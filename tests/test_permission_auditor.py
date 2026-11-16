"""Tests for cross-file Frappe/source permission gathering, Guest escalation, and clean reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from indexer.permission_auditor import audit_repository_permissions_and_security
from mcp_server.code_index_server import CodeIndexPipeline
from pr_bot.analyzer import analyze_pr_directories, analyze_repository
from pr_bot.cli import main as cli_main
from pr_bot.gemini_reviewer import (
    build_review_prompt,
    classify_custom_guard_helpers,
    verify_review_claims,
)


def test_frappe_doctype_guest_and_all_permissions_flagged(tmp_path: Path) -> None:
    """DocType JSONs and Custom DocPerm fixtures exposing Guest/All permissions are flagged."""
    dt_dir = tmp_path / "custom_app" / "doctype" / "portal_note"
    dt_dir.mkdir(parents=True)
    (dt_dir / "portal_note.json").write_text(
        json.dumps(
            {
                "doctype": "DocType",
                "name": "Portal Note",
                "permissions": [
                    {"role": "Guest", "read": 1, "write": 1, "export": 1},
                    {"role": "System Manager", "read": 1, "write": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    fixtures = tmp_path / "custom_app" / "fixtures"
    fixtures.mkdir()
    (fixtures / "custom_docperm.json").write_text(
        json.dumps(
            [
                {
                    "doctype": "Custom DocPerm",
                    "parent": "Sales Invoice",
                    "role": "All",
                    "read": 1,
                    "delete": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    report = analyze_repository(tmp_path)
    high_msgs = [f.message for f in report.findings if f.severity == "HIGH"]
    assert any("Guest" in m and "write" in m and "Portal Note" in m for m in high_msgs)
    assert any("All" in m and "delete" in m and "Sales Invoice" in m for m in high_msgs)
    assert len(report.permission_inventory) >= 2

    md = report.to_markdown()
    assert "### Findings" in md
    assert "### Permission & Privilege Inventory" in md
    assert "Portal Note" in md
    assert "`:1`" not in md


def test_frappe_multi_hop_guest_escalation_and_guarded_webhook(tmp_path: Path) -> None:
    """Unguarded allow_guest endpoint reaching ignore_permissions across files flags HIGH."""
    app = tmp_path / "my_app"
    app.mkdir()
    (app / "service.py").write_text(
        "def _elevate_and_save(docname: str) -> None:\n"
        "    doc = frappe.get_doc('Invoice', docname)\n"
        "    doc.save(ignore_permissions=True)\n\n"
        "def verify_signature(sig: str) -> bool:\n"
        "    return sig == 'ok'\n",
        encoding="utf-8",
    )
    (app / "api.py").write_text(
        "from my_app.service import _elevate_and_save, verify_signature\n\n"
        "@frappe.whitelist(allow_guest=True)\n"
        "def public_approve(docname: str) -> None:\n"
        "    return _elevate_and_save(docname)\n\n"
        "@frappe.whitelist(allow_guest=True)\n"
        "def webhook_handler(sig: str, docname: str) -> None:\n"
        "    if not verify_signature(sig):\n"
        "        return\n"
        "    return _elevate_and_save(docname)\n",
        encoding="utf-8",
    )

    report = analyze_repository(tmp_path)
    escalations = [f for f in report.findings if f.category == "GUEST_PRIVILEGE_ESCALATION"]
    flagged_syms = {f.symbol_name for f in escalations}
    assert "public_approve" in flagged_syms
    assert "webhook_handler" not in flagged_syms
    assert any("_elevate_and_save" in f.message for f in escalations)


def test_frappe_class_method_sql_injection_hooks_and_dead_code(tmp_path: Path) -> None:
    """Class method security checks work via dedent, and hooks/lifecycle avoid DEAD_CODE FP."""
    app = tmp_path / "billing_app"
    app.mkdir()
    (app / "hooks.py").write_text(
        'doc_events = {"Sales Order": {"on_update": "billing_app.events.handle_so_update"}}\n',
        encoding="utf-8",
    )
    (app / "events.py").write_text(
        "class CustomInvoice:\n"
        "    def validate(self) -> None:\n"
        "        frappe.db.sql(f'SELECT * FROM tabUser WHERE name = {self.owner}')\n"
        "        frappe.db.commit()\n\n"
        "def handle_so_update(doc) -> None:\n"
        "    frappe.set_user('Administrator')\n"
        "    return None\n\n"
        "@frappe.whitelist()\n"
        "def lookup_secret(token):\n"
        "    return frappe.db.get_value('TokenDoc', {'token': token}, 'name')\n\n"
        "def _never_called_helper(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y * 2\n",
        encoding="utf-8",
    )

    report = analyze_repository(tmp_path)
    cats = {(f.category, f.symbol_name) for f in report.findings}
    assert ("SQL_INJECTION", "validate") in cats
    assert ("ATOMICITY_RISK", "validate") in cats
    assert ("PRIVILEGE_ESCALATION", "handle_so_update") in cats
    assert ("SECURITY_RISK", "lookup_secret") in cats
    assert ("DEAD_CODE", "_never_called_helper") in cats
    # Lifecycle & hooks.py methods must NOT be misreported as DEAD_CODE
    assert ("DEAD_CODE", "validate") not in cats
    assert ("DEAD_CODE", "handle_so_update") not in cats


def test_pr_mode_flags_newly_added_guest_permission_and_clean_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR analysis flags newly added Guest bypasses and emits clean AI-first Markdown artifact."""
    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()

    (base / "api.py").write_text(
        "@frappe.whitelist()\ndef get_item(name: str) -> str:\n    return name\n",
        encoding="utf-8",
    )
    (head / "api.py").write_text(
        "@frappe.whitelist(allow_guest=True)\n"
        "def get_item(name: str) -> object:\n"
        "    return frappe.get_all('Salary Slip', filters={'employee': name})\n\n"
        "def _unused_temp_helper() -> int:\n"
        "    tmp = 1\n"
        "    return tmp\n",
        encoding="utf-8",
    )
    diff = (
        "diff --git a/api.py b/api.py\n"
        "--- a/api.py\n"
        "+++ b/api.py\n"
        "@@ -1,3 +1,3 @@\n"
    )
    report = analyze_pr_directories(base, head, diff_text=diff)
    assert any(
        f.severity == "HIGH" and f.category == "GUEST_PERMISSION" and f.symbol_name == "get_item"
        for f in report.findings
    )
    md = report.to_markdown()
    assert "| Severity | Category | Location | Symbol | Details |" in md
    assert "<details>" in md  # Informational findings tucked inside collapsible details

    # Verify CLI places Gemini review above deterministic report when available
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "pr_bot.cli.generate_vertex_gemini_review",
        lambda **kwargs: "1. **High Risk:** `api.py:1` exposes `Salary Slip` to Guest.",
    )
    out_md = tmp_path / "repo_report.md"
    rc = cli_main(["--mode", "repo", "--repo", str(head), "--out-md", str(out_md)])
    assert rc == 0
    artifact_text = out_md.read_text(encoding="utf-8")
    assert artifact_text.startswith("### Gemini Review")
    assert "## Astraea Periodic" in artifact_text


def test_frappe_synthetic_framework_edges_and_deep_witness_path(tmp_path: Path) -> None:
    """4-hop path across doc.submit() -> hooks.py doc_events -> flags.ignore_permissions."""
    app = tmp_path / "erp_ext"
    dt = app / "doctype" / "sales_invoice"
    dt.mkdir(parents=True)
    (dt / "sales_invoice.json").write_text(
        json.dumps(
            {
                "doctype": "DocType",
                "name": "Sales Invoice",
                "permissions": [{"role": "Accounts User", "read": 1, "write": 1}],
            }
        ),
        encoding="utf-8",
    )
    (app / "hooks.py").write_text(
        'doc_events = {"Sales Invoice": {"on_submit": "erp_ext.billing.sync_erp"}}\n',
        encoding="utf-8",
    )
    (app / "utils.py").write_text(
        "def log_metric(name: str) -> None:\n    pass\n",
        encoding="utf-8",
    )
    (app / "ledger.py").write_text(
        "def post_gl_override(entry: str) -> None:\n"
        "    entry_doc = frappe.get_doc('GL Entry', entry)\n"
        "    entry_doc.flags.ignore_permissions = True\n"
        "    entry_doc.insert()\n",
        encoding="utf-8",
    )
    (app / "billing.py").write_text(
        "from erp_ext.ledger import post_gl_override\n\n"
        "def sync_erp(doc, method=None) -> None:\n"
        "    if doc.grand_total > 0:\n"
        "        post_gl_override(doc.name)\n",
        encoding="utf-8",
    )
    (app / "api.py").write_text(
        "from erp_ext.utils import log_metric\n\n"
        "@frappe.whitelist(allow_guest=True)\n"
        "def guest_submit_invoice(inv_name: str) -> None:\n"
        "    log_metric('submit')\n"
        "    doc = frappe.get_doc('Sales Invoice', inv_name)\n"
        "    doc.submit()\n",
        encoding="utf-8",
    )

    report = analyze_repository(tmp_path)
    assert len(report.witness_paths) >= 1
    wp = next(
        w
        for w in report.witness_paths
        if w["source_symbol"] == "guest_submit_invoice" and w["sink_symbol"] == "post_gl_override"
    )
    step_names = [s["symbol_name"] for s in wp["steps"]]
    assert step_names == ["guest_submit_invoice", "sync_erp", "post_gl_override"]
    # Unrelated 1-hop neighbor `log_metric` must NOT appear in the witness path or prompt
    assert "log_metric" not in step_names
    assert "Sales Invoice" in wp["target_doctype_perms"]

    prompt = build_review_prompt(report)
    assert "Unsanitized Source-to-Sink Witness Paths" in prompt
    assert "doc.grand_total > 0" in prompt
    assert "def log_metric" not in prompt
    assert "utils.py" not in prompt


def test_cached_bottom_up_guard_summary_prunes_witness_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom helper classified once as AUTH_GUARD is cached in SQLite and prunes CPG path."""
    from google import genai

    app = tmp_path / "shop"
    app.mkdir()
    (app / "auth.py").write_text(
        "def check_merchant_hmac(payload: str) -> None:\n"
        "    if not payload:\n"
        "        raise ValueError('bad')\n",
        encoding="utf-8",
    )
    (app / "db_ops.py").write_text(
        "def force_write(name: str) -> None:\n"
        "    frappe.db.set_value('Order', name, 'status', 'Paid')\n",
        encoding="utf-8",
    )
    (app / "api.py").write_text(
        "from shop.auth import check_merchant_hmac\n"
        "from shop.db_ops import force_write\n\n"
        "@frappe.whitelist(allow_guest=True)\n"
        "def webhook(payload: str, name: str) -> None:\n"
        "    check_merchant_hmac(payload)\n"
        "    return force_write(name)\n",
        encoding="utf-8",
    )

    # Mock helper -> sink call edge from check_merchant_hmac to force_write?
    # Note: here check_merchant_hmac is a direct helper if it wraps force_write:
    (app / "auth.py").write_text(
        "from shop.db_ops import force_write\n\n"
        "def gate_and_write(payload: str, name: str) -> None:\n"
        "    if payload == 'valid':\n"
        "        return force_write(name)\n",
        encoding="utf-8",
    )
    (app / "api.py").write_text(
        "from shop.auth import gate_and_write\n\n"
        "@frappe.whitelist(allow_guest=True)\n"
        "def webhook(payload: str, name: str) -> None:\n"
        "    return gate_and_write(payload, name)\n",
        encoding="utf-8",
    )

    pipe = CodeIndexPipeline(storage_dir=tmp_path / ".idx")
    pipe.index_workspace(tmp_path)

    res1 = audit_repository_permissions_and_security(
        tmp_path, pipe.store.units.values(), pipe.cpg, pipe.symbol_to_unit
    )
    assert len(res1.witness_paths) == 1
    assert any(h["symbol_name"] == "gate_and_write" for h in res1.candidate_helpers)

    llm_calls = {"n": 0}

    class _FakeModels:
        def generate_content(self, **kwargs: Any) -> Any:
            llm_calls["n"] += 1
            return type(
                "Resp",
                (),
                {"text": '{"role": "AUTH_GUARD", "rationale": "validates payload"}'},
            )()

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = _FakeModels()

    monkeypatch.setattr(genai, "Client", _FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    summaries = classify_custom_guard_helpers(res1.candidate_helpers, store=pipe.store)
    assert llm_calls["n"] == 1
    # Second classification uses SQLite cache with 0 LLM calls
    summaries_cached = classify_custom_guard_helpers(res1.candidate_helpers, store=pipe.store)
    assert llm_calls["n"] == 1
    assert summaries_cached == summaries

    # Re-running audit with cached summaries prunes the witness path cleanly
    res2 = audit_repository_permissions_and_security(
        tmp_path,
        pipe.store.units.values(),
        pipe.cpg,
        pipe.symbol_to_unit,
        cached_summaries=summaries_cached,
    )
    assert res2.witness_paths == []
    pipe.store.close()


def test_false_security_claims_detected_in_source_and_hooks(tmp_path: Path) -> None:
    """No-op verify_* guards, always-True has_permission hooks, and guest doc contradictions."""
    app = tmp_path / "billing_app"
    app.mkdir()
    (app / "hooks.py").write_text(
        'has_permission = {"Invoice": "billing_app.perms.always_allow_perm"}\n',
        encoding="utf-8",
    )
    (app / "perms.py").write_text(
        "def always_allow_perm(doc, ptype, user) -> bool:\n"
        "    return True\n\n"
        "def verify_webhook_signature(sig: str) -> bool:\n"
        "    return True\n",
        encoding="utf-8",
    )
    (app / "api.py").write_text(
        "from billing_app.perms import verify_webhook_signature\n\n"
        "@frappe.whitelist(allow_guest=True)\n"
        "def guest_pay(order_id: str, sig: str) -> None:\n"
        '    """Admin only endpoint requiring authentication."""\n'
        "    verify_webhook_signature(sig)\n"
        "    frappe.db.set_value('Order', order_id, 'status', 'Paid')\n",
        encoding="utf-8",
    )

    report = analyze_repository(tmp_path)
    false_claims = [f for f in report.findings if f.category == "FALSE_SECURITY_CLAIM"]
    flagged_syms = {f.symbol_name for f in false_claims}
    assert "always_allow_perm" in flagged_syms
    assert "verify_webhook_signature" in flagged_syms
    assert "guest_pay" in flagged_syms
    # Calling the fake verify_webhook_signature guard must NOT hide the Guest escalation
    assert any(
        f.category == "GUEST_PERMISSION" and f.symbol_name == "guest_pay"
        for f in report.findings
    )


def test_post_llm_false_claim_verification_and_guard_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage-4 AST guard check rejects fake LLM AUTH_GUARD; post-LLM gate flags hallucinations."""
    from google import genai

    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()
    (base / "api.py").write_text(
        "def public_write(doc_name: str) -> None:\n    pass\n",
        encoding="utf-8",
    )
    (head / "api.py").write_text(
        "@frappe.whitelist(allow_guest=True)\n"
        "def public_write(doc_name: str) -> None:\n"
        "    return frappe.db.set_value('Order', doc_name, 'status', 'Paid')\n",
        encoding="utf-8",
    )
    diff = (
        "diff --git a/api.py b/api.py\n--- a/api.py\n+++ b/api.py\n@@ -1,2 +1,3 @@\n"
    )
    pr_report = analyze_pr_directories(base, head, diff_text=diff)

    # 1. Stage-4 guard verification rejects hallucinated AUTH_GUARD on a pass-through helper
    class _FakeModels:
        def generate_content(self, **kwargs: Any) -> Any:
            return type(
                "Resp",
                (),
                {"text": '{"role": "AUTH_GUARD", "rationale": "validates caller token"}'},
            )()

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = _FakeModels()

    monkeypatch.setattr(genai, "Client", _FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    verdicts = classify_custom_guard_helpers(
        [
            {
                "content_hash": "abc123",
                "symbol_name": "passthrough_write",
                "source_text": (
                    "def passthrough_write(doc_name: str) -> None:\n"
                    "    return frappe.db.set_value('Order', doc_name, 'status', 'Paid')\n"
                ),
            }
        ]
    )
    role, rationale = verdicts["abc123"]
    assert role == "NEUTRAL"
    assert "Rejected false AUTH_GUARD claim" in rationale

    # 2. Grounded review passes without false claims
    clean_text, clean_claims = verify_review_claims(
        "1. `api.py:1` exposes `public_write` to Guest without authorization.", pr_report
    )
    assert clean_claims == []
    assert "[UNVERIFIED CLAIM" not in clean_text

    # 3. Hallucinated review is caught across missing file, bad line, missing sym, NFC, and safety
    hallucinated = (
        "1. `ghost_module.py:42` leaks tokens.\n"
        "2. `api.py:999` calls `phantom_guard()` cleanly.\n"
        "3. `public_write` is NFC Verified with no security issues."
    )
    verified, detected = verify_review_claims(hallucinated, pr_report)
    claim_types = {c["type"] for c in detected}
    assert {
        "MISSING_FILE",
        "OUT_OF_RANGE_LINE",
        "UNKNOWN_SYMBOL",
        "NFC_CONTRADICTION",
        "SAFETY_CONTRADICTION",
    }.issubset(claim_types)
    assert "### Claim Verification Gate" in verified
    assert verified.count("[UNVERIFIED CLAIM:") == 3


def test_100k_token_budget_packs_more_than_five_hotspots_and_clones(tmp_path: Path) -> None:
    """Default 100k-token review prompt includes >5 hotspots, clone clusters, and full-plan ask."""
    for idx in range(8):
        (tmp_path / f"mod_{idx}.py").write_text(
            f"def complex_handler_{idx}(a: int, b: int) -> int:\n"
            "    total = 0\n"
            "    if a > 0:\n"
            "        total += a\n"
            "    if b > 0:\n"
            "        total += b\n"
            "    if a > b:\n"
            "        total += 1\n"
            "    return total\n",
            encoding="utf-8",
        )

    report = analyze_repository(tmp_path, complexity_threshold=3)
    assert len(report.hotspot_sources) == 8

    prompt_100k = build_review_prompt(report)
    for idx in range(8):
        assert f"complex_handler_{idx}" in prompt_100k
    assert "### Semantic Clone Clusters to Consolidate" in prompt_100k
    assert "P0 Security" in prompt_100k
    assert "not limited to 3 items" in prompt_100k

    # Tight explicit token cap bounds prompt size gracefully without error
    prompt_compact = build_review_prompt(report, max_prompt_tokens=600)
    assert "### Deterministic Analysis Report" in prompt_compact
    assert len(prompt_compact) < len(prompt_100k)

