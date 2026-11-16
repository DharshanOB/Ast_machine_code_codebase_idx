"""Cross-file Frappe & general permission, guest-escalation, and security auditor."""

from __future__ import annotations

import ast
import json
import os
import re
import textwrap
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from indexer.ast_chunker import MAX_FILE_BYTES
from indexer.models import ASTChunkNode, CPGEdgeKind, Language, UnifiedCodeUnit
from indexer.semantic_cpg import SemanticCPGBuilder

_AUDITOR_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "build",
        "dist",
        "target",
        "venv",
        ".venv",
        "__pycache__",
        "vendor",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

_FRAPPE_LIFECYCLE_METHODS: frozenset[str] = frozenset(
    {
        "autoname",
        "before_naming",
        "before_validate",
        "validate",
        "before_save",
        "before_insert",
        "after_insert",
        "on_update",
        "on_update_after_submit",
        "before_submit",
        "on_submit",
        "before_cancel",
        "on_cancel",
        "on_trash",
        "after_delete",
        "before_rename",
        "after_rename",
        "get_context",
        "get_list_context",
        "has_website_permission",
        "has_permission",
        "get_permission_query_conditions",
        "onload",
    }
)

_HIGH_DOC_PERMS: tuple[str, ...] = (
    "write",
    "create",
    "delete",
    "submit",
    "cancel",
    "amend",
    "export",
    "import",
    "email",
    "print",
    "report",
    "share",
)

_WORKFLOW_WRITE_RE = re.compile(
    r"^\s*(write-all|[a-z-]+:\s*write)\s*(?:#.*)?$",
    re.MULTILINE,
)
_CPP_DANGEROUS_CALL_RE = re.compile(r"\b(gets|strcpy|strcat|sprintf|vsprintf|system|popen)\s*\(")
_JS_DANGEROUS_RE = re.compile(r"\b(eval|Function)\s*\(|child_process\.exec\s*\(|\.innerHTML\s*=")
_CONTAINER_AST_KINDS: frozenset[str] = frozenset(
    {
        "class_definition",
        "class_declaration",
        "class_specifier",
        "struct_specifier",
        "impl_item",
        "struct_item",
        "trait_item",
        "enum_specifier",
    }
)


def _is_test_path(rel_file: str) -> bool:
    norm = rel_file.replace("\\", "/").lstrip("./")
    parts = norm.split("/")
    fname = parts[-1] if parts else norm
    return "tests" in parts or "test" in parts[:-1] or fname.startswith("test_") or fname.endswith(
        "_test.py"
    )


def _is_ignored_dir(name: str) -> bool:
    return name in _AUDITOR_IGNORED_DIRS or name.startswith(".")


def _rel_path(path_str: str | Path, root: Path) -> str:
    try:
        return str(Path(path_str).resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path_str).replace("\\", "/")


def _iter_repo_files(root: Path, *, include_dot_github: bool = False) -> Iterable[Path]:
    """Yield regular files under root while pruning heavy/ignored directories in-place."""
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not (
                d in _AUDITOR_IGNORED_DIRS
                or (d.startswith(".") and not (include_dot_github and d == ".github"))
            )
        )
        base = Path(dirpath)
        for fname in sorted(filenames):
            fpath = base / fname
            try:
                if fpath.is_file() and fpath.stat().st_size <= MAX_FILE_BYTES:
                    yield fpath
            except OSError:
                continue


def _call_dotted_name(node: ast.AST) -> str:
    """Resolve an AST call target expression into a dotted string like `frappe.db.sql`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _kwarg_truthy(call: ast.Call, kw_name: str) -> bool:
    """Return True when a keyword argument is statically truthy (`True` or non-zero int)."""
    for kw in call.keywords:
        if kw.arg != kw_name:
            continue
        if isinstance(kw.value, ast.Constant):
            return bool(kw.value.value)
    return False


def _is_dynamic_sql_arg(expr: ast.AST) -> bool:
    """Return True when an expression constructs SQL via f-string, `%`, `+`, or `.format`."""
    if isinstance(expr, ast.JoinedStr):
        return True
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Mod)):
        return True
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "format"
    ):
        return True
    return False


def _whitelist_decorator_info(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, bool]:
    """Return `(is_whitelisted, allow_guest)` for a Python function AST node."""
    is_whitelisted = False
    allow_guest = False
    for dec in fn_node.decorator_list:
        name = ""
        if isinstance(dec, ast.Call):
            name = _call_dotted_name(dec.func)
            if name in ("frappe.whitelist", "whitelist"):
                is_whitelisted = True
                if _kwarg_truthy(dec, "allow_guest"):
                    allow_guest = True
        else:
            name = _call_dotted_name(dec)
            if name in ("frappe.whitelist", "whitelist"):
                is_whitelisted = True
    return is_whitelisted, allow_guest


def _extract_param_names(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return non-self/cls parameter names accepted by a Python function."""
    all_args = [
        *fn_node.args.posonlyargs,
        *fn_node.args.args,
        *fn_node.args.kwonlyargs,
    ]
    names = {a.arg for a in all_args if a.arg not in ("self", "cls")}
    return names


def _has_type_guard(fn_node: ast.FunctionDef | ast.AsyncFunctionDef, param_names: set[str]) -> bool:
    """Return True if function performs `isinstance(...)`, `cint(...)`, or `flt(...)` checks."""
    if not param_names:
        return True
    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Call):
            cname = _call_dotted_name(sub.func)
            if cname in ("isinstance", "cint", "flt", "frappe.utils.cint", "frappe.utils.flt"):
                return True
    return False


def _has_try_finally_restore(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if any `try ... finally` block calls `set_user` in its cleanup handler."""
    for sub in ast.walk(fn_node):
        if isinstance(sub, ast.Try) and sub.finalbody:
            for stmt in sub.finalbody:
                for inner in ast.walk(stmt):
                    if isinstance(inner, ast.Call) and _call_dotted_name(inner.func).endswith(
                        "set_user"
                    ):
                        return True
    return False


def _scan_doctype_and_fixture_permissions(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Scan DocType JSONs, Custom DocPerm, and Property Setter fixtures for permissions."""
    findings: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    doctype_perms: dict[str, list[dict[str, Any]]] = {}

    for fpath in _iter_repo_files(root):
        if fpath.suffix.lower() != ".json":
            continue
        rel = _rel_path(fpath, root)
        try:
            raw_text = fpath.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(raw_text)
        except (OSError, json.JSONDecodeError):
            continue

        records: list[dict[str, Any]] = []
        doctype_name = ""
        if isinstance(data, dict):
            if data.get("doctype") == "DocType" and isinstance(data.get("permissions"), list):
                doctype_name = str(data.get("name") or fpath.stem)
                records = [p for p in data["permissions"] if isinstance(p, dict)]
                doctype_perms[doctype_name] = records
        elif isinstance(data, list):
            for rec in data:
                if not isinstance(rec, dict):
                    continue
                if rec.get("doctype") == "Custom DocPerm":
                    records.append(rec)
                elif rec.get("doctype") == "Property Setter" and rec.get("doc_type"):
                    prop = str(rec.get("property", ""))
                    val = str(rec.get("value", "")).strip()
                    if prop in _HIGH_DOC_PERMS and val == "1":
                        target_dt = str(rec.get("doc_type"))
                        msg = (
                            f"Property Setter enables `{prop}=1` on `{target_dt}`; "
                            "verify role scoping."
                        )
                        findings.append(
                            {
                                "severity": "MEDIUM",
                                "category": "UNWANTED_PERMISSION",
                                "file_path": rel,
                                "line": 1,
                                "symbol_name": target_dt,
                                "message": msg,
                            }
                        )
            if records:
                doctype_name = str(records[0].get("parent", fpath.stem))
                doctype_perms.setdefault(doctype_name, []).extend(records)

        for perm in records:
            role = str(perm.get("role", "")).strip()
            if role not in ("Guest", "All"):
                continue
            elevated = [flag for flag in _HIGH_DOC_PERMS if int(perm.get(flag, 0) or 0) == 1]
            has_read = int(perm.get("read", 0) or 0) == 1
            if not elevated and not has_read:
                continue
            perm_summary = ", ".join([*(["read"] if has_read else []), *elevated])
            target_dt = str(perm.get("parent") or doctype_name or fpath.stem)

            if role == "Guest":
                severity = "HIGH" if elevated else "MEDIUM"
                category = "GUEST_PERMISSION"
                msg = (
                    f"Role 'Guest' has elevated DocType permission(s) [{perm_summary}] "
                    f"on DocType `{target_dt}`."
                )
            elif elevated:
                severity = "HIGH"
                category = "UNWANTED_PERMISSION"
                msg = (
                    f"Role 'All' has elevated DocType permission(s) [{perm_summary}] "
                    f"on DocType `{target_dt}`."
                )
            else:
                continue

            inventory.append(
                {
                    "file_path": rel,
                    "line": 1,
                    "symbol_name": target_dt,
                    "role_or_scope": f"DocType ({role})",
                    "permission": perm_summary,
                    "severity": severity,
                    "reason": msg,
                }
            )
            findings.append(
                {
                    "severity": severity,
                    "category": category,
                    "file_path": rel,
                    "line": 1,
                    "symbol_name": target_dt,
                    "message": msg,
                }
            )

    return findings, inventory, doctype_perms


def _extract_strings_from_ast(node: ast.AST) -> list[str]:
    """Collect dotted string literals (hook targets) from an AST value expression."""
    out: list[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        val = node.value.strip()
        if "." in val and " " not in val:
            out.append(val)
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for elt in node.elts:
            out.extend(_extract_strings_from_ast(elt))
    elif isinstance(node, ast.Dict):
        for val_node in node.values:
            out.extend(_extract_strings_from_ast(val_node))
    return out


@dataclass
class _HooksSpec:
    doc_events: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    override_whitelisted: dict[str, str] = field(default_factory=dict)
    override_doctype_class: dict[str, str] = field(default_factory=dict)
    has_permission_dotted: set[str] = field(default_factory=set)
    has_permission_symbols: set[str] = field(default_factory=set)
    hooked_dotted: set[str] = field(default_factory=set)
    hooked_symbols: set[str] = field(default_factory=set)
    findings: list[dict[str, Any]] = field(default_factory=list)


def _eval_hook_string_list(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        val = node.value.strip()
        return [val] if val else []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: list[str] = []
        for elt in node.elts:
            out.extend(_eval_hook_string_list(elt))
        return out
    return []


def _parse_frappe_hooks(root: Path) -> _HooksSpec:
    """Extract structured Frappe hooks wiring (`doc_events`, overrides, perm hooks)."""
    spec = _HooksSpec()
    target_vars = {
        "doc_events",
        "override_whitelisted_methods",
        "override_doctype_class",
        "permission_query_conditions",
        "has_permission",
        "scheduler_events",
        "website_route_rules",
    }

    for fpath in _iter_repo_files(root):
        if fpath.name != "hooks.py":
            continue
        rel = _rel_path(fpath, root)
        try:
            tree = ast.parse(fpath.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if not (isinstance(tgt, ast.Name) and tgt.id in target_vars):
                    continue
                for dotted in _extract_strings_from_ast(node.value):
                    spec.hooked_dotted.add(dotted)
                    spec.hooked_symbols.add(dotted.rsplit(".", 1)[-1])

                if tgt.id == "doc_events" and isinstance(node.value, ast.Dict):
                    for dt_k, dt_v in zip(node.value.keys, node.value.values, strict=True):
                        if not (isinstance(dt_k, ast.Constant) and isinstance(dt_k.value, str)):
                            continue
                        if not isinstance(dt_v, ast.Dict):
                            continue
                        ev_map = spec.doc_events.setdefault(dt_k.value.strip(), {})
                        for ev_k, ev_v in zip(dt_v.keys, dt_v.values, strict=True):
                            if isinstance(ev_k, ast.Constant) and isinstance(ev_k.value, str):
                                ev_map.setdefault(ev_k.value.strip(), []).extend(
                                    _eval_hook_string_list(ev_v)
                                )
                elif (
                    tgt.id in ("override_whitelisted_methods", "override_doctype_class")
                    and isinstance(node.value, ast.Dict)
                ):
                    dst_map = (
                        spec.override_whitelisted
                        if tgt.id == "override_whitelisted_methods"
                        else spec.override_doctype_class
                    )
                    for k_node, v_node in zip(node.value.keys, node.value.values, strict=True):
                        if (
                            isinstance(k_node, ast.Constant)
                            and isinstance(k_node.value, str)
                            and isinstance(v_node, ast.Constant)
                            and isinstance(v_node.value, str)
                        ):
                            dst_map[k_node.value.strip()] = v_node.value.strip()

                if tgt.id in ("permission_query_conditions", "has_permission"):
                    if tgt.id == "has_permission":
                        for dotted in _extract_strings_from_ast(node.value):
                            spec.has_permission_dotted.add(dotted)
                            spec.has_permission_symbols.add(dotted.rsplit(".", 1)[-1])
                    spec.findings.append(
                        {
                            "severity": "INFO",
                            "category": "PERMISSION_HOOK",
                            "file_path": rel,
                            "line": node.lineno,
                            "symbol_name": tgt.id,
                            "message": (
                                f"Custom `{tgt.id}` hook registered; verify it handles "
                                "`Guest` / restricted users without always returning True."
                            ),
                        }
                    )

    return spec


_CLAIMED_AUTH_DOC_RE = re.compile(
    r"\b(?:admin\s+only|requires?\s+(?:auth(?:entication)?|login|admin|role|permission)|"
    r"authenticated\s+users?\s+only|must\s+be\s+logged\s+in|restricted\s+to\s+authorized)\b",
    re.IGNORECASE,
)
_GUARD_NAME_PREFIXES: tuple[str, ...] = (
    "verify_",
    "check_perm",
    "require_",
    "authorize_",
    "validate_perm",
    "ensure_auth",
)
_AUTH_SIGNAL_TOKENS: tuple[str, ...] = (
    "has_permission",
    "only_for",
    "check_permission",
    "hmac",
    "compare_digest",
    "verify",
    "signature",
    "session",
    "get_roles",
    "token",
    "secret",
    "password",
    "authorize",
)
_SANITIZER_SIGNAL_TOKENS: tuple[str, ...] = (
    "isinstance",
    "cint",
    "flt",
    "fullmatch",
    "match",
    "escape_html",
    "strip_html",
    "int",
    "str",
    "float",
    "bool",
)


def _strip_docstring(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """Return function body statements excluding any leading string docstring."""
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(stmts[0].value, ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        return stmts[1:]
    return stmts


def _extract_fn_security_signals(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, bool]:
    """Summarize AST enforcement, branching, and return behavior for a Python function."""
    effective = _strip_docstring(list(fn_node.body))
    returns = [n for n in ast.walk(fn_node) if isinstance(n, ast.Return)]
    calls = [_call_dotted_name(n.func) for n in ast.walk(fn_node) if isinstance(n, ast.Call)]
    has_if = any(isinstance(n, (ast.If, ast.IfExp)) for n in ast.walk(fn_node))
    has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(fn_node))
    has_throw = any(
        c.endswith((".throw", "abort", "sys.exit")) or c == "throw" for c in calls
    )
    only_true_return = bool(returns) and all(
        isinstance(r.value, ast.Constant) and r.value.value is True for r in returns
    )
    top_level_true = any(
        isinstance(s, ast.Return) and isinstance(s.value, ast.Constant) and s.value.value is True
        for s in effective
    )
    top_level_noop = not effective or all(isinstance(s, ast.Pass) for s in effective)
    src_dump = ast.dump(fn_node).lower()
    has_auth_sig = any(tok in src_dump for tok in _AUTH_SIGNAL_TOKENS)
    has_in_check = any(
        isinstance(n, ast.Assert)
        or (isinstance(n, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops))
        for n in ast.walk(fn_node)
    )
    has_sanitizer_sig = has_in_check or any(
        c.rsplit(".", 1)[-1] in _SANITIZER_SIGNAL_TOKENS for c in calls
    )
    return {
        "is_noop": top_level_noop,
        "only_true_return": only_true_return,
        "unconditional_true": top_level_true,
        "has_if": has_if,
        "has_rejection": has_raise or has_throw,
        "has_auth_signal": has_auth_sig,
        "has_sanitizer_signal": has_sanitizer_sig,
    }


def verify_helper_classification_ast(source_text: str, claimed_role: str) -> tuple[bool, str]:
    """Deterministically verify whether an LLM `AUTH_GUARD`/`SANITIZER` label matches the AST."""
    role = claimed_role.upper()
    if role not in ("AUTH_GUARD", "SANITIZER"):
        return True, ""
    try:
        tree = ast.parse(textwrap.dedent(source_text))
    except SyntaxError:
        return True, ""
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not fns:
        return False, "No function definition found to substantiate security role."
    sig = _extract_fn_security_signals(fns[0])

    if role == "AUTH_GUARD":
        if sig["is_noop"] or sig["only_true_return"] or sig["unconditional_true"]:
            return False, "Function is a no-op or unconditionally returns True."
        if not (sig["has_if"] or sig["has_rejection"] or sig["has_auth_signal"]):
            return False, "No branching check, exception/throw, or auth signal found in AST."
    elif role == "SANITIZER":
        if sig["is_noop"] or sig["only_true_return"]:
            return False, "Function is a no-op or unconditionally returns True."
        if not (sig["has_sanitizer_signal"] or sig["has_rejection"]):
            return False, "No type guard, pattern check, or rejection path found in AST."
    return True, ""


def _detect_false_security_claims(
    chunk: ASTChunkNode,
    rel_file: str,
    hooks_spec: _HooksSpec,
    has_auth_guard: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Detect misleading/no-op auth guards, fake `has_permission` hooks, and doc contradictions."""
    findings: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    is_fake_guard = False

    if chunk.language != Language.PYTHON:
        return findings, inventory, False
    try:
        tree = ast.parse(textwrap.dedent(chunk.source_text))
    except SyntaxError:
        return findings, inventory, False

    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not fns:
        return findings, inventory, False
    top_fn = fns[0]
    sym = chunk.symbol_name or top_fn.name
    _, allow_guest = _whitelist_decorator_info(top_fn)
    sig = _extract_fn_security_signals(top_fn)

    is_perm_hook = (
        top_fn.name in hooks_spec.has_permission_symbols
        or any(d.endswith(f".{top_fn.name}") for d in hooks_spec.has_permission_dotted)
    )
    looks_like_guard = top_fn.name.startswith(_GUARD_NAME_PREFIXES) or is_perm_hook

    if looks_like_guard and (
        sig["is_noop"] or sig["only_true_return"] or sig["unconditional_true"]
    ):
        is_fake_guard = True
        hook_note = " (`has_permission` hook)" if is_perm_hook else ""
        msg = (
            f"Security helper `{sym}`{hook_note} is a no-op or unconditionally returns True "
            "without enforcing authorization checks."
        )
        findings.append(
            {
                "severity": "HIGH",
                "category": "FALSE_SECURITY_CLAIM",
                "file_path": rel_file,
                "line": chunk.line_start,
                "symbol_name": sym,
                "message": msg,
            }
        )
        inventory.append(
            {
                "file_path": rel_file,
                "line": chunk.line_start,
                "symbol_name": sym,
                "role_or_scope": "False Security Claim",
                "permission": "No-op / always-True guard",
                "severity": "HIGH",
                "reason": msg,
            }
        )
    elif looks_like_guard and not (sig["has_if"] or sig["has_rejection"] or sig["has_auth_signal"]):
        is_fake_guard = True
        msg = (
            f"Function `{sym}` claims authorization/verification by name but performs "
            "no conditional check, exception raise, or auth check."
        )
        findings.append(
            {
                "severity": "MEDIUM",
                "category": "FALSE_SECURITY_CLAIM",
                "file_path": rel_file,
                "line": chunk.line_start,
                "symbol_name": sym,
                "message": msg,
            }
        )

    if allow_guest and not has_auth_guard and _CLAIMED_AUTH_DOC_RE.search(chunk.source_text):
        msg = (
            f"Endpoint `{sym}` claims restricted/authenticated access in comments or docstring "
            "but declares `@frappe.whitelist(allow_guest=True)` with no auth guard."
        )
        findings.append(
            {
                "severity": "HIGH",
                "category": "FALSE_SECURITY_CLAIM",
                "file_path": rel_file,
                "line": chunk.line_start,
                "symbol_name": sym,
                "message": msg,
            }
        )

    return findings, inventory, is_fake_guard


def _inspect_python_chunk(
    chunk: ASTChunkNode,
    rel_file: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    bool,
    bool,
    bool,
    list[str],
]:
    """Inspect a Python AST chunk for permissions, bypasses, guards, and security risks."""
    findings: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    is_guest_entry = False
    has_auth_guard = False
    privileged_sinks: list[str] = []

    try:
        tree = ast.parse(textwrap.dedent(chunk.source_text))
    except SyntaxError:
        return findings, inventory, False, False, False, []

    base_offset = max(0, chunk.line_start - 1)
    fn_nodes = [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    top_fn = fn_nodes[0] if fn_nodes else None
    is_whitelisted, allow_guest = _whitelist_decorator_info(top_fn) if top_fn else (False, False)
    sym = chunk.symbol_name or (top_fn.name if top_fn else "<module>")
    param_names = _extract_param_names(top_fn) if top_fn else set()
    has_type_check = _has_type_guard(top_fn, param_names) if top_fn else True

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    for sub in calls:
        cname = _call_dotted_name(sub.func)
        if any(
            g in cname
            for g in (
                "has_permission",
                "only_for",
                "check_permission",
                "verify_signature",
                "hmac.compare_digest",
                "validate_webhook",
            )
        ):
            has_auth_guard = True

    if allow_guest:
        is_guest_entry = True
        inventory.append(
            {
                "file_path": rel_file,
                "line": chunk.line_start,
                "symbol_name": sym,
                "role_or_scope": "Python Endpoint (Guest)",
                "permission": "@frappe.whitelist(allow_guest=True)",
                "severity": "MEDIUM",
                "reason": "Publicly callable without authentication.",
            }
        )
    elif sym == "get_context" and "frappe" in chunk.source_text:
        is_guest_entry = True

    saw_untyped_orm_filter = False
    saw_guest_mutation_finding = False

    # Detect attribute assignments such as `doc.flags.ignore_permissions = True`
    for assign in (n for n in ast.walk(tree) if isinstance(n, ast.Assign)):
        val_is_truthy = isinstance(assign.value, ast.Constant) and bool(assign.value.value)
        if not val_is_truthy:
            continue
        for tgt in assign.targets:
            tname = _call_dotted_name(tgt)
            if tname.endswith(".ignore_permissions") or tname.endswith(
                ".ignore_user_permissions"
            ):
                line_no = base_offset + getattr(assign, "lineno", 1)
                sink_label = f"{tname}=True"
                privileged_sinks.append(sink_label)
                sev = "HIGH" if allow_guest else "MEDIUM"
                msg = (
                    f"`{sink_label}` bypasses Frappe permission checks"
                    + (" inside an `allow_guest=True` endpoint." if allow_guest else ".")
                )
                inventory.append(
                    {
                        "file_path": rel_file,
                        "line": line_no,
                        "symbol_name": sym,
                        "role_or_scope": "Permission Bypass",
                        "permission": sink_label,
                        "severity": sev,
                        "reason": msg,
                    }
                )
                findings.append(
                    {
                        "severity": sev,
                        "category": "GUEST_PERMISSION" if allow_guest else "PERMISSION_BYPASS",
                        "file_path": rel_file,
                        "line": line_no,
                        "symbol_name": sym,
                        "message": msg,
                    }
                )
                if allow_guest:
                    saw_guest_mutation_finding = True

    for sub in calls:
        cname = _call_dotted_name(sub.func)
        line_no = base_offset + getattr(sub, "lineno", 1)

        # Permission bypass keyword flags
        for bypass_kw in ("ignore_permissions", "ignore_user_permissions"):
            if _kwarg_truthy(sub, bypass_kw):
                sink_label = f"{cname}({bypass_kw}=True)"
                privileged_sinks.append(sink_label)
                sev = "HIGH" if allow_guest else "MEDIUM"
                msg = (
                    f"`{sink_label}` bypasses Frappe permission checks"
                    + (" inside an `allow_guest=True` endpoint." if allow_guest else ".")
                )
                inventory.append(
                    {
                        "file_path": rel_file,
                        "line": line_no,
                        "symbol_name": sym,
                        "role_or_scope": "Permission Bypass",
                        "permission": sink_label,
                        "severity": sev,
                        "reason": msg,
                    }
                )
                findings.append(
                    {
                        "severity": sev,
                        "category": "GUEST_PERMISSION" if allow_guest else "PERMISSION_BYPASS",
                        "file_path": rel_file,
                        "line": line_no,
                        "symbol_name": sym,
                        "message": msg,
                    }
                )
                if allow_guest:
                    saw_guest_mutation_finding = True

        # Privilege escalation via frappe.set_user("Administrator")
        if cname.endswith("set_user") and sub.args:
            arg0 = sub.args[0]
            if isinstance(arg0, ast.Constant) and str(arg0.value) == "Administrator":
                privileged_sinks.append('set_user("Administrator")')
                restores = top_fn is not None and _has_try_finally_restore(top_fn)
                sev = "HIGH" if (allow_guest or not restores) else "MEDIUM"
                msg = (
                    '`frappe.set_user("Administrator")` escalates session to System Admin'
                    + ("" if restores else " without a `try/finally` user restore.")
                )
                inventory.append(
                    {
                        "file_path": rel_file,
                        "line": line_no,
                        "symbol_name": sym,
                        "role_or_scope": "Privilege Escalation",
                        "permission": 'set_user("Administrator")',
                        "severity": sev,
                        "reason": msg,
                    }
                )
                findings.append(
                    {
                        "severity": sev,
                        "category": "PRIVILEGE_ESCALATION",
                        "file_path": rel_file,
                        "line": line_no,
                        "symbol_name": sym,
                        "message": msg,
                    }
                )

        # frappe.get_all vs get_list in whitelisted endpoints
        if cname in ("frappe.get_all", "frappe.db.get_all"):
            privileged_sinks.append(cname)
            if is_whitelisted:
                sev = "HIGH" if allow_guest else "MEDIUM"
                msg = (
                    f"`{cname}` bypasses user/document permissions inside "
                    + ("an `allow_guest=True` endpoint" if allow_guest else "a whitelisted method")
                    + "; prefer `frappe.get_list`."
                )
                findings.append(
                    {
                        "severity": sev,
                        "category": "GUEST_PERMISSION" if allow_guest else "PERMISSION_BYPASS",
                        "file_path": rel_file,
                        "line": line_no,
                        "symbol_name": sym,
                        "message": msg,
                    }
                )
                saw_guest_mutation_finding = saw_guest_mutation_finding or allow_guest

        # Direct DB mutations inside allow_guest=True
        if cname in (
            "frappe.db.set_value",
            "frappe.db.delete",
            "frappe.get_doc",
            "frappe.new_doc",
            "frappe.delete_doc",
        ):
            privileged_sinks.append(cname)
            if allow_guest and not has_auth_guard and not saw_guest_mutation_finding:
                saw_guest_mutation_finding = True
                findings.append(
                    {
                        "severity": "HIGH",
                        "category": "GUEST_PERMISSION",
                        "file_path": rel_file,
                        "line": line_no,
                        "symbol_name": sym,
                        "message": (
                            f"`allow_guest=True` endpoint invokes `{cname}` without an explicit "
                            "permission or signature check."
                        ),
                    }
                )

        # Type-confusion check on whitelisted ORM queries
        if (
            is_whitelisted
            and not has_type_check
            and not saw_untyped_orm_filter
            and cname
            in ("frappe.db.get_value", "frappe.db.get_list", "frappe.get_list", "frappe.db.exists")
        ):
            used_param = any(
                isinstance(n, ast.Name) and n.id in param_names
                for arg in [*sub.args, *(k.value for k in sub.keywords)]
                for n in ast.walk(arg)
            )
            if used_param:
                saw_untyped_orm_filter = True
                findings.append(
                    {
                        "severity": "MEDIUM",
                        "category": "SECURITY_RISK",
                        "file_path": rel_file,
                        "line": line_no,
                        "symbol_name": sym,
                        "message": (
                            f"Whitelisted `{cname}` uses parameter(s) without an "
                            "`isinstance(..., str)` type check (risk of ORM list-filter bypass)."
                        ),
                    }
                )

        # Raw SQL string interpolation
        if cname == "frappe.db.sql" and sub.args and _is_dynamic_sql_arg(sub.args[0]):
            privileged_sinks.append("frappe.db.sql(dynamic)")
            findings.append(
                {
                    "severity": "HIGH",
                    "category": "SQL_INJECTION",
                    "file_path": rel_file,
                    "line": line_no,
                    "symbol_name": sym,
                    "message": (
                        "Dynamic SQL interpolation in `frappe.db.sql`; use parameterized `%s` "
                        "placeholders or `frappe.qb`."
                    ),
                }
            )

        # Mid-transaction commit/rollback
        if cname in ("frappe.db.commit", "frappe.db.rollback"):
            findings.append(
                {
                    "severity": "MEDIUM",
                    "category": "ATOMICITY_RISK",
                    "file_path": rel_file,
                    "line": line_no,
                    "symbol_name": sym,
                    "message": (
                        f"Explicit `{cname}()` mid-request breaks transaction atomicity and "
                        "can persist partial state."
                    ),
                }
            )

        # General Python security sinks
        if cname in ("eval", "exec"):
            privileged_sinks.append(cname)
            findings.append(
                {
                    "severity": "HIGH",
                    "category": "SECURITY_RISK",
                    "file_path": rel_file,
                    "line": line_no,
                    "symbol_name": sym,
                    "message": f"Dangerous dynamic code execution `{cname}()`.",
                }
            )
        elif cname.startswith("subprocess.") and _kwarg_truthy(sub, "shell"):
            privileged_sinks.append(f"{cname}(shell=True)")
            findings.append(
                {
                    "severity": "HIGH",
                    "category": "SECURITY_RISK",
                    "file_path": rel_file,
                    "line": line_no,
                    "symbol_name": sym,
                    "message": f"`{cname}(..., shell=True)` permits shell injection.",
                }
            )
        elif cname in ("os.chmod", "chmod") and len(sub.args) >= 2:
            mode_arg = sub.args[1]
            if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, int):
                if (mode_arg.value & 0o002) != 0:
                    perm_oct = oct(mode_arg.value)
                    inventory.append(
                        {
                            "file_path": rel_file,
                            "line": line_no,
                            "symbol_name": sym,
                            "role_or_scope": "Filesystem Mode",
                            "permission": perm_oct,
                            "severity": "HIGH",
                            "reason": f"World-writable file mode `{perm_oct}` passed to `{cname}`.",
                        }
                    )
                    findings.append(
                        {
                            "severity": "HIGH",
                            "category": "UNWANTED_PERMISSION",
                            "file_path": rel_file,
                            "line": line_no,
                            "symbol_name": sym,
                            "message": f"World-writable file permission `{perm_oct}` in `{cname}`.",
                        }
                    )

    if allow_guest and not saw_guest_mutation_finding and not has_auth_guard:
        findings.append(
            {
                "severity": "MEDIUM",
                "category": "GUEST_PERMISSION",
                "file_path": rel_file,
                "line": chunk.line_start,
                "symbol_name": sym,
                "message": (
                    "`@frappe.whitelist(allow_guest=True)` exposes this endpoint to "
                    "unauthenticated Guest users; confirm intent and add validation/rate limiting."
                ),
            }
        )

    return findings, inventory, is_guest_entry, has_auth_guard, is_whitelisted, privileged_sinks


def _scan_workflow_and_container_permissions(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan GitHub Actions workflows and Dockerfiles for overly broad permissions."""
    findings: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []

    for fpath in _iter_repo_files(root, include_dot_github=True):
        rel = _rel_path(fpath, root)
        if ".github/workflows/" in rel and fpath.suffix.lower() in (".yml", ".yaml"):
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in _WORKFLOW_WRITE_RE.finditer(text):
                perm = m.group(1).strip()
                line_no = text[: m.start()].count("\n") + 1
                sev = "HIGH" if perm == "write-all" else "INFO"
                if perm == "write-all":
                    findings.append(
                        {
                            "severity": "HIGH",
                            "category": "UNWANTED_PERMISSION",
                            "file_path": rel,
                            "line": line_no,
                            "symbol_name": fpath.name,
                            "message": (
                                "GitHub workflow grants `write-all` token permissions; "
                                "scope to explicit least-privilege keys."
                            ),
                        }
                    )
                inventory.append(
                    {
                        "file_path": rel,
                        "line": line_no,
                        "symbol_name": fpath.name,
                        "role_or_scope": "CI Workflow Token",
                        "permission": perm,
                        "severity": sev,
                        "reason": "Workflow write permission grant.",
                    }
                )
        elif fpath.name == "Dockerfile" or fpath.name.startswith("Dockerfile."):
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            has_user = any(
                ln.strip().upper().startswith("USER ") and not ln.strip().endswith("root")
                for ln in text.splitlines()
            )
            if "FROM " in text and not has_user:
                msg = "Dockerfile has no non-root `USER` directive (container runs as root)."
                inventory.append(
                    {
                        "file_path": rel,
                        "line": 1,
                        "symbol_name": fpath.name,
                        "role_or_scope": "Container User",
                        "permission": "USER root (default)",
                        "severity": "MEDIUM",
                        "reason": msg,
                    }
                )
                findings.append(
                    {
                        "severity": "MEDIUM",
                        "category": "UNWANTED_PERMISSION",
                        "file_path": rel,
                        "line": 1,
                        "symbol_name": fpath.name,
                        "message": msg,
                    }
                )

    return findings, inventory


_METHOD_TO_FRAPPE_EVENTS: dict[str, tuple[str, ...]] = {
    "insert": (
        "before_insert",
        "before_validate",
        "validate",
        "before_save",
        "after_insert",
        "on_update",
    ),
    "save": ("before_validate", "validate", "before_save", "on_update"),
    "submit": (
        "before_validate",
        "validate",
        "before_save",
        "before_submit",
        "on_submit",
        "on_update",
    ),
    "cancel": ("before_cancel", "on_cancel"),
    "delete": ("on_trash", "after_delete"),
}


def _normalize_doctype_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _extract_doc_lifecycle_invocations(chunk: ASTChunkNode) -> list[tuple[str, str, int]]:
    """Extract `(doctype_name, lifecycle_method, line)` pairs from Frappe doc calls."""
    if chunk.language != Language.PYTHON:
        return []
    try:
        tree = ast.parse(textwrap.dedent(chunk.source_text))
    except SyntaxError:
        return []

    base_offset = max(0, chunk.line_start - 1)
    var_to_doctype: dict[str, str] = {}
    invocations: list[tuple[str, str, int]] = []

    def _doctype_from_get_doc(call: ast.Call) -> str | None:
        cname = _call_dotted_name(call.func)
        if cname not in ("frappe.get_doc", "frappe.new_doc", "get_doc", "new_doc"):
            return None
        if call.args:
            arg0 = call.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                return arg0.value.strip()
            if isinstance(arg0, ast.Dict):
                for k_n, v_n in zip(arg0.keys, arg0.values, strict=True):
                    if (
                        isinstance(k_n, ast.Constant)
                        and k_n.value == "doctype"
                        and isinstance(v_n, ast.Constant)
                        and isinstance(v_n.value, str)
                    ):
                        return v_n.value.strip()
        for kw in call.keywords:
            if kw.arg == "doctype" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    return kw.value.value.strip()
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            dt = _doctype_from_get_doc(node.value)
            if dt:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        var_to_doctype[tgt.id] = dt
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method not in _METHOD_TO_FRAPPE_EVENTS:
                continue
            dt_name: str | None = None
            if isinstance(node.func.value, ast.Call):
                dt_name = _doctype_from_get_doc(node.func.value)
            elif isinstance(node.func.value, ast.Name):
                dt_name = var_to_doctype.get(node.func.value.id)
            if dt_name:
                line_no = base_offset + getattr(node, "lineno", 1)
                invocations.append((dt_name, method, line_no))

    return invocations


def _extract_branch_conditions(chunk: ASTChunkNode) -> list[str]:
    """Extract concise `if` condition expressions from a Python function AST."""
    if chunk.language != Language.PYTHON:
        return []
    try:
        tree = ast.parse(textwrap.dedent(chunk.source_text))
    except SyntaxError:
        return []
    conds: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            try:
                cond_text = ast.unparse(node.test).strip()
            except Exception:
                continue
            if cond_text and cond_text not in conds:
                conds.append(cond_text[:120])
    return conds[:5]


def _add_framework_edge(
    cpg: SemanticCPGBuilder,
    src_uri: str,
    dst_uri: str,
    *,
    line: int,
    label: str,
) -> None:
    """Inject a deduplicated synthetic FRAMEWORK_CALLS edge into CPG."""
    if src_uri == dst_uri or src_uri not in cpg.graph or dst_uri not in cpg.graph:
        return
    for _, existing_dst, data in cpg.graph.out_edges(src_uri, data=True):
        if (
            existing_dst == dst_uri
            and data.get("kind") == CPGEdgeKind.FRAMEWORK_CALLS.value
            and data.get("metadata", {}).get("framework_event") == label
        ):
            return
    cpg._add_edge(
        src_uri,
        dst_uri,
        CPGEdgeKind.FRAMEWORK_CALLS,
        line=line,
        metadata={"framework_event": label},
        confidence="confirmed",
        evidence=f"frappe_framework:{label}",
    )


def _inject_frappe_framework_edges(
    units: list[UnifiedCodeUnit],
    hooks_spec: _HooksSpec,
    cpg: SemanticCPGBuilder,
) -> None:
    """Stage 1: Augment Astraea's CPG with Frappe lifecycle, doc_events, and override edges."""
    dotted_to_uris: dict[str, list[str]] = {}
    name_to_uris: dict[str, list[str]] = {}
    class_method_uris: dict[tuple[str, str], list[str]] = {}

    for uri, sym_node in cpg.symbols.items():
        if sym_node.kind in _CONTAINER_AST_KINDS:
            continue
        name_to_uris.setdefault(sym_node.display_name, []).append(uri)

    for unit in units:
        ast_chunk = unit.ast_chunk
        if unit.scip_symbol is None or ast_chunk.node_kind in _CONTAINER_AST_KINDS:
            continue
        uri = unit.scip_symbol.symbol_uri
        # Map `<module>.<symbol>` dotted suffixes cleanly from file_path
        norm_mod = ast_chunk.file_path.replace("\\", "/").removesuffix(".py").replace("/", ".")
        parts = [p for p in norm_mod.split(".") if p]
        for tail_idx in range(len(parts)):
            dotted = ".".join([*parts[tail_idx:], ast_chunk.symbol_name])
            dotted_to_uris.setdefault(dotted, []).append(uri)

        # Extract enclosing class name from scope_header (`class_definition SalesInvoice`)
        for hdr_line in ast_chunk.scope_header.splitlines():
            stripped = hdr_line.strip()
            if stripped.startswith("class_definition "):
                cls_name = stripped.removeprefix("class_definition ").strip()
                class_method_uris.setdefault(
                    (_normalize_doctype_key(cls_name), ast_chunk.symbol_name), []
                ).append(uri)
                # Register override_doctype_class alias if this class overrides a DocType
                for dt_name, override_cls_dotted in hooks_spec.override_doctype_class.items():
                    override_short = override_cls_dotted.rsplit(".", 1)[-1]
                    if cls_name == override_short:
                        class_method_uris.setdefault(
                            (_normalize_doctype_key(dt_name), ast_chunk.symbol_name), []
                        ).append(uri)

    # Wire override_whitelisted_methods into callers
    for orig_dotted, override_dotted in hooks_spec.override_whitelisted.items():
        orig_short = orig_dotted.rsplit(".", 1)[-1]
        override_short = override_dotted.rsplit(".", 1)[-1]
        target_uris = dotted_to_uris.get(override_dotted) or name_to_uris.get(override_short, [])
        if not target_uris:
            continue
        orig_uris = dotted_to_uris.get(orig_dotted) or name_to_uris.get(orig_short, [])
        for orig_uri in orig_uris:
            for caller_uri, _, data in list(cpg.graph.in_edges(orig_uri, data=True)):
                if data.get("kind") == CPGEdgeKind.CALLS.value:
                    for tgt_uri in target_uris:
                        _add_framework_edge(
                            cpg,
                            caller_uri,
                            tgt_uri,
                            line=int(data.get("line") or 1),
                            label=f"override:{orig_short}",
                        )
            for tgt_uri in target_uris:
                _add_framework_edge(
                    cpg, orig_uri, tgt_uri, line=1, label=f"override:{orig_short}"
                )

    # Wire doc.insert()/save()/submit()/cancel()/delete() -> controller lifecycle & hooks.doc_events
    for unit in units:
        if unit.scip_symbol is None or unit.ast_chunk.node_kind in _CONTAINER_AST_KINDS:
            continue
        caller_uri = unit.scip_symbol.symbol_uri
        for dt_name, method_name, call_line in _extract_doc_lifecycle_invocations(unit.ast_chunk):
            events = _METHOD_TO_FRAPPE_EVENTS.get(method_name, ())
            dt_key = _normalize_doctype_key(dt_name)
            for ev in events:
                for ctrl_uri in class_method_uris.get((dt_key, ev), []):
                    _add_framework_edge(
                        cpg, caller_uri, ctrl_uri, line=call_line, label=f"{dt_name}.{ev}"
                    )
                for hook_dt_key in (dt_name, "*"):
                    for handler_dotted in hooks_spec.doc_events.get(hook_dt_key, {}).get(ev, []):
                        handler_uris = dotted_to_uris.get(handler_dotted) or name_to_uris.get(
                            handler_dotted.rsplit(".", 1)[-1], []
                        )
                        for h_uri in handler_uris:
                            _add_framework_edge(
                                cpg,
                                caller_uri,
                                h_uri,
                                line=call_line,
                                label=f"doc_events:{dt_name}.{ev}",
                            )


@dataclass
class PermissionAuditResult:
    """Structured output from the 4-stage neuro-symbolic permission & security audit."""

    findings: list[dict[str, Any]]
    inventory: list[dict[str, Any]]
    witness_paths: list[dict[str, Any]]
    candidate_helpers: list[dict[str, Any]]


def _build_bidirectional_witness_path(
    src_uri: str,
    sink_uri: str,
    edge_kinds: frozenset[str],
    cpg: SemanticCPGBuilder,
    sanitizer_uris: set[str],
    max_depth: int = 8,
) -> list[tuple[str, str]] | None:
    """Bidirectional BFS returning shortest unsanitized `(uri, incoming_edge_label)` spine."""
    if src_uri == sink_uri or src_uri in sanitizer_uris:
        return None
    if src_uri not in cpg.graph or sink_uri not in cpg.graph:
        return None

    fwd_parent: dict[str, tuple[str | None, str]] = {src_uri: (None, "SOURCE")}
    bwd_child: dict[str, tuple[str | None, str]] = {sink_uri: (None, "")}
    fwd_q: deque[tuple[str, int]] = deque([(src_uri, 0)])
    bwd_q: deque[tuple[str, int]] = deque([(sink_uri, 0)])
    meet_node: str | None = None

    while fwd_q and bwd_q:
        if len(fwd_q) <= len(bwd_q):
            curr, depth = fwd_q.popleft()
            if depth < max_depth:
                for _, nxt, data in cpg.graph.out_edges(curr, data=True):
                    kind = str(data.get("kind", ""))
                    if kind not in edge_kinds or nxt in sanitizer_uris:
                        continue
                    ev = str((data.get("metadata") or {}).get("framework_event", ""))
                    edge_label = f"{kind}:{ev}" if ev else kind
                    if nxt not in fwd_parent:
                        fwd_parent[nxt] = (curr, edge_label)
                        if nxt in bwd_child:
                            meet_node = nxt
                            break
                        fwd_q.append((nxt, depth + 1))
        else:
            curr, depth = bwd_q.popleft()
            if depth < max_depth:
                for prev_u, _, data in cpg.graph.in_edges(curr, data=True):
                    kind = str(data.get("kind", ""))
                    if kind not in edge_kinds or prev_u in sanitizer_uris:
                        continue
                    ev = str((data.get("metadata") or {}).get("framework_event", ""))
                    edge_label = f"{kind}:{ev}" if ev else kind
                    if prev_u not in bwd_child:
                        bwd_child[prev_u] = (curr, edge_label)
                        if prev_u in fwd_parent:
                            meet_node = prev_u
                            break
                        bwd_q.append((prev_u, depth + 1))
        if meet_node is not None:
            break

    if meet_node is None:
        return None

    prefix: list[tuple[str, str]] = []
    cur: str | None = meet_node
    while cur is not None:
        parent_u, lbl = fwd_parent[cur]
        prefix.append((cur, lbl))
        cur = parent_u
    prefix.reverse()

    suffix: list[tuple[str, str]] = []
    next_u, next_lbl = bwd_child[meet_node]
    while next_u is not None:
        suffix.append((next_u, next_lbl))
        next_u, next_lbl = bwd_child[next_u]

    return prefix + suffix


def _trace_witness_paths(
    guest_units: list[tuple[UnifiedCodeUnit, str, bool]],
    unit_sinks_by_uid: dict[str, list[str]],
    guard_uris: set[str],
    cpg: SemanticCPGBuilder,
    symbol_to_unit: dict[str, str],
    units_by_uid: dict[str, UnifiedCodeUnit],
    doctype_perms: dict[str, list[dict[str, Any]]],
    cached_summaries: dict[str, tuple[str, str]],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stage 3: Bidirectional Source->Sink shortest-path slicing with cached-guard pruning."""
    sanitizer_uris: set[str] = set(guard_uris)
    for uri, uid in symbol_to_unit.items():
        u = units_by_uid.get(uid)
        if u is not None:
            summary = cached_summaries.get(u.ast_chunk.chunk_id)
            if summary is not None and summary[0] in ("AUTH_GUARD", "SANITIZER"):
                sanitizer_uris.add(uri)

    allowed_edge_kinds = frozenset(
        {CPGEdgeKind.CALLS.value, CPGEdgeKind.FRAMEWORK_CALLS.value}
    )
    sink_uris = [
        uri for uri, uid in symbol_to_unit.items() if uid in unit_sinks_by_uid
    ]

    escalations: list[dict[str, Any]] = []
    witness_paths: list[dict[str, Any]] = []
    candidate_helpers_by_id: dict[str, dict[str, Any]] = {}

    for unit, rel_file, has_guard in guest_units:
        if has_guard or unit.scip_symbol is None:
            continue
        start_uri = unit.scip_symbol.symbol_uri
        if start_uri in sanitizer_uris:
            continue
        src_name = unit.ast_chunk.symbol_name or "<guest_endpoint>"

        for sink_uri in sink_uris:
            sink_uid = symbol_to_unit.get(sink_uri, "")
            if not sink_uid or sink_uid == unit.unit_id:
                continue
            spine = _build_bidirectional_witness_path(
                start_uri,
                sink_uri,
                allowed_edge_kinds,
                cpg,
                sanitizer_uris,
                max_depth=8,
            )
            if not spine or len(spine) < 2:
                continue

            sinks = unit_sinks_by_uid.get(sink_uid, [])
            sink_str = ", ".join(dict.fromkeys(sinks))
            step_dicts: list[dict[str, Any]] = []
            labels_for_msg: list[str] = []
            referenced_doctypes: list[str] = []

            for step_uri, edge_label in spine:
                sym_node = cpg.symbols.get(step_uri)
                step_uid = symbol_to_unit.get(step_uri, "")
                step_unit = units_by_uid.get(step_uid)
                step_sym = sym_node.display_name if sym_node else step_uri
                step_file = _rel_path(sym_node.file_path, root) if sym_node else ""
                step_line = sym_node.line_start if sym_node else 1
                branch_conds = (
                    _extract_branch_conditions(step_unit.ast_chunk) if step_unit else []
                )
                step_src = step_unit.ast_chunk.source_text[:1500] if step_unit else ""

                if ":" in edge_label:
                    ev_part = edge_label.rsplit(":", 1)[-1]
                    for dt_name in doctype_perms:
                        if ev_part.startswith(f"{dt_name}.") and dt_name not in referenced_doctypes:
                            referenced_doctypes.append(dt_name)
                if step_unit is not None:
                    for dt_name, _, _ in _extract_doc_lifecycle_invocations(step_unit.ast_chunk):
                        if dt_name in doctype_perms and dt_name not in referenced_doctypes:
                            referenced_doctypes.append(dt_name)

                prefix = f"[{edge_label}] -> " if edge_label != "SOURCE" else ""
                labels_for_msg.append(f"{prefix}{step_file}:{step_sym}")
                step_dicts.append(
                    {
                        "symbol_name": step_sym,
                        "file_path": step_file,
                        "line_start": step_line,
                        "edge_kind": edge_label,
                        "branch_conditions": branch_conds,
                        "source_text": step_src,
                    }
                )
                if (
                    step_unit is not None
                    and step_uri not in (start_uri, sink_uri)
                    and step_unit.ast_chunk.chunk_id not in cached_summaries
                ):
                    candidate_helpers_by_id[step_unit.ast_chunk.chunk_id] = {
                        "content_hash": step_unit.ast_chunk.chunk_id,
                        "symbol_name": step_sym,
                        "file_path": step_file,
                        "source_text": step_unit.ast_chunk.source_text[:2000],
                    }

            sink_step = step_dicts[-1]
            spine_str = " ".join(labels_for_msg)
            escalations.append(
                {
                    "severity": "HIGH",
                    "category": "GUEST_PRIVILEGE_ESCALATION",
                    "file_path": rel_file,
                    "line": unit.ast_chunk.line_start,
                    "symbol_name": src_name,
                    "message": (
                        f"Unsanitized {len(spine) - 1}-hop path to [{sink_str}]: {spine_str}."
                    ),
                }
            )
            witness_paths.append(
                {
                    "source_symbol": src_name,
                    "source_file": rel_file,
                    "sink_symbol": sink_step["symbol_name"],
                    "sink_file": sink_step["file_path"],
                    "sinks": sinks,
                    "hops": len(spine) - 1,
                    "steps": step_dicts,
                    "target_doctype_perms": {
                        dt: doctype_perms[dt] for dt in referenced_doctypes
                    },
                }
            )

    return escalations, witness_paths, list(candidate_helpers_by_id.values())


def audit_repository_permissions_and_security(
    repo_root: Path,
    units: Iterable[UnifiedCodeUnit],
    cpg: SemanticCPGBuilder,
    symbol_to_unit: dict[str, str],
    cached_summaries: dict[str, tuple[str, str]] | None = None,
) -> PermissionAuditResult:
    """Run 4-stage Frappe & security audit across ASTs, DocType JSONs, hooks, and CPG."""
    root = repo_root.resolve()
    unit_list = list(units)
    units_by_uid = {u.unit_id: u for u in unit_list}
    findings: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []

    dt_findings, dt_inv, doctype_perms = _scan_doctype_and_fixture_permissions(root)
    findings.extend(dt_findings)
    inventory.extend(dt_inv)

    hooks_spec = _parse_frappe_hooks(root)
    findings.extend(hooks_spec.findings)

    # Stage 1: Inject synthetic Frappe lifecycle, doc_events, and override edges into CPG.
    _inject_frappe_framework_edges(unit_list, hooks_spec, cpg)

    wf_findings, wf_inv = _scan_workflow_and_container_permissions(root)
    findings.extend(wf_findings)
    inventory.extend(wf_inv)

    guest_units: list[tuple[UnifiedCodeUnit, str, bool]] = []
    unit_sinks_by_uid: dict[str, list[str]] = {}
    guard_uris: set[str] = set()
    fake_guard_uris: set[str] = set()
    fake_guard_names: set[str] = set()
    whitelisted_uids: set[str] = set()

    for unit in unit_list:
        ast_chunk = unit.ast_chunk
        if ast_chunk.node_kind in _CONTAINER_AST_KINDS:
            continue
        rel_file = _rel_path(ast_chunk.file_path, root)
        if _is_test_path(rel_file):
            continue
        if ast_chunk.language == Language.PYTHON:
            u_findings, u_inv, is_guest, has_guard, is_wl, sinks = _inspect_python_chunk(
                ast_chunk, rel_file
            )
            fc_findings, fc_inv, is_fake_guard = _detect_false_security_claims(
                ast_chunk, rel_file, hooks_spec, has_guard
            )
            findings.extend(u_findings)
            findings.extend(fc_findings)
            inventory.extend(u_inv)
            inventory.extend(fc_inv)
            if is_fake_guard:
                if ast_chunk.symbol_name:
                    fake_guard_names.add(ast_chunk.symbol_name)
                if unit.scip_symbol is not None:
                    fake_guard_uris.add(unit.scip_symbol.symbol_uri)
            elif has_guard and unit.scip_symbol is not None:
                guard_uris.add(unit.scip_symbol.symbol_uri)
            if is_wl:
                whitelisted_uids.add(unit.unit_id)
            if sinks:
                unit_sinks_by_uid[unit.unit_id] = sinks
            if is_guest:
                guest_units.append((unit, rel_file, has_guard and not is_fake_guard))
        elif ast_chunk.language in (Language.C, Language.CPP):
            for m in _CPP_DANGEROUS_CALL_RE.finditer(ast_chunk.source_text):
                line_no = ast_chunk.line_start + ast_chunk.source_text[: m.start()].count("\n")
                findings.append(
                    {
                        "severity": "MEDIUM",
                        "category": "SECURITY_RISK",
                        "file_path": rel_file,
                        "line": line_no,
                        "symbol_name": ast_chunk.symbol_name or "<c_fn>",
                        "message": f"Unsafe C/C++ call `{m.group(1)}()`.",
                    }
                )
        elif ast_chunk.language == Language.JAVASCRIPT:
            for m in _JS_DANGEROUS_RE.finditer(ast_chunk.source_text):
                line_no = ast_chunk.line_start + ast_chunk.source_text[: m.start()].count("\n")
                findings.append(
                    {
                        "severity": "HIGH",
                        "category": "SECURITY_RISK",
                        "file_path": rel_file,
                        "line": line_no,
                        "symbol_name": ast_chunk.symbol_name or "<js_fn>",
                        "message": f"Unsafe JavaScript sink `{m.group(0).strip()}`.",
                    }
                )

    guard_uris -= fake_guard_uris
    if fake_guard_names:
        guest_units = [
            (
                u,
                rf,
                False
                if hg and any(fn in u.ast_chunk.source_text for fn in fake_guard_names)
                else hg,
            )
            for u, rf, hg in guest_units
        ]

    escalations, witness_paths, candidate_helpers = _trace_witness_paths(
        guest_units=guest_units,
        unit_sinks_by_uid=unit_sinks_by_uid,
        guard_uris=guard_uris,
        cpg=cpg,
        symbol_to_unit=symbol_to_unit,
        units_by_uid=units_by_uid,
        doctype_perms=doctype_perms,
        cached_summaries=cached_summaries or {},
        root=root,
    )
    findings.extend(escalations)

    # Identify unreferenced private functions while honoring Frappe lifecycle, hooks, & overrides.
    for unit in unit_list:
        ast_chunk = unit.ast_chunk
        if ast_chunk.node_kind in _CONTAINER_AST_KINDS or ast_chunk.has_syntax_error:
            continue
        rel_file = _rel_path(ast_chunk.file_path, root)
        if _is_test_path(rel_file) or unit.scip_symbol is None:
            continue
        sym = ast_chunk.symbol_name
        if not sym or sym.startswith("__") or sym in ("main",):
            continue
        if (
            unit.unit_id in whitelisted_uids
            or sym in _FRAPPE_LIFECYCLE_METHODS
            or sym in hooks_spec.hooked_symbols
            or any(h.endswith(f".{sym}") for h in hooks_spec.hooked_dotted)
        ):
            continue
        uri = unit.scip_symbol.symbol_uri
        if uri not in cpg.graph:
            continue
        incoming_calls = [
            1
            for _, _, data in cpg.graph.in_edges(uri, data=True)
            if str(data.get("kind", ""))
            in (CPGEdgeKind.CALLS.value, CPGEdgeKind.FRAMEWORK_CALLS.value)
        ]
        if (
            not incoming_calls
            and (sym.startswith("_") and not sym.startswith("__"))
            and (ast_chunk.line_end - ast_chunk.line_start + 1) >= 3
        ):
            findings.append(
                {
                    "severity": "INFO",
                    "category": "DEAD_CODE",
                    "file_path": rel_file,
                    "line": ast_chunk.line_start,
                    "symbol_name": sym,
                    "message": f"Private function `{sym}` has 0 incoming CPG callers.",
                }
            )

    return PermissionAuditResult(
        findings=findings,
        inventory=inventory,
        witness_paths=witness_paths,
        candidate_helpers=candidate_helpers,
    )
