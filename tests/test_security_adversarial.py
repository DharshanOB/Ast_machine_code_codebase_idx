"""Security, adversarial repository, incremental indexing, and store consistency tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from indexer.llvm_ir_pass import LLVMIRPass, resolve_clang_binary
from mcp_server.code_index_server import CodeIndexPipeline
from pr_bot.cli import main as cli_main
from storage.hybrid_store import HybridCodeStore


def test_hostile_compile_commands_strips_plugin_and_response_files(tmp_path: Path) -> None:
    """Hostile compile_commands.json flags (-fplugin, @rsp) must be stripped before clang."""
    src = tmp_path / "safe.c"
    src.write_text("int ok(int x) { return x + 1; }\n", encoding="utf-8")

    cc_path = tmp_path / "compile_commands.json"
    cc_path.write_text(
        json.dumps(
            [
                {
                    "directory": str(tmp_path),
                    "file": str(src),
                    "arguments": [
                        "clang",
                        "-fplugin=/tmp/pwn.so",
                        "@/etc/passwd",
                        "-Xclang",
                        "-load",
                        "-O2",
                        "-DTEST_FLAG=1",
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    pipeline = CodeIndexPipeline()
    summary = pipeline.index_workspace(tmp_path, compile_commands_path=cc_path)
    assert summary["indexed_units"] == 1


def test_malformed_and_binary_files_do_not_crash_pipeline(tmp_path: Path) -> None:
    """Malformed syntax, null bytes, and deep nesting must degrade gracefully."""
    (tmp_path / "broken.py").write_text("def f(:\n    ((((\n", encoding="utf-8")
    (tmp_path / "nulls.c").write_bytes(b"int bad() { return \x00\xff\xfe; }\n")
    (tmp_path / "deep.c").write_text(
        "#define M(x) (x + 1)\n"
        "int deep(int a) {\n"
        + ("  a = M(a);\n" * 50)
        + "  return a;\n}\n",
        encoding="utf-8",
    )

    pipeline = CodeIndexPipeline()
    summary = pipeline.index_workspace(tmp_path)
    assert summary["indexed_units"] >= 2


def test_compiler_timeout_handling(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify compiler subprocess timeouts are caught and sanitized without crashing."""
    if resolve_clang_binary() is None:
        return
    src = tmp_path / "slow.c"
    src.write_text("int slow(int x) { return x; }\n", encoding="utf-8")

    def _raise_timeout(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="clang", timeout=15)

    monkeypatch.setattr(LLVMIRPass, "_run_ir_compiler", staticmethod(_raise_timeout))
    pipeline = CodeIndexPipeline()
    summary = pipeline.index_workspace(tmp_path)
    assert summary["degraded_units"] == 1
    inspected = pipeline.inspect_lowering_pipeline("slow")
    assert inspected["degraded_reason"] == "Compiler subprocess timed out"


def test_untrusted_cli_and_mcp_inputs(tmp_path: Path) -> None:
    """Nonexistent paths and oversized outputs must be handled safely."""
    pipeline = CodeIndexPipeline()
    res = pipeline.index_workspace(tmp_path / "does_not_exist")
    assert res["indexed_units"] == 0
    assert "error" in res

    out_md = tmp_path / "out.md"
    out_json = tmp_path / "out.json"
    rc = cli_main(
        [
            "--mode",
            "repo",
            "--repo",
            str(tmp_path),
            "--out-md",
            str(out_md),
            "--out-comments",
            str(out_json),
        ]
    )
    assert rc == 0


def test_hybrid_store_reindexing_deletion_and_sqlite_recovery(tmp_path: Path) -> None:
    """Verify Tantivy/Qdrant/SQLite stay consistent across updates, deletes, and restarts."""
    storage_dir = tmp_path / "idx"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    py_file = repo_dir / "mod.py"
    py_file.write_text("def alpha(x: int) -> int:\n    return x + 1\n", encoding="utf-8")

    pipeline = CodeIndexPipeline(storage_dir=storage_dir)
    first = pipeline.index_workspace(repo_dir)
    assert first["indexed_units"] == 1
    assert pipeline.store.verify_consistency()["consistent"] is True

    # Re-run incremental with no changes: skips unchanged file
    second = pipeline.index_workspace(repo_dir, incremental=True)
    assert second["indexed_units"] == 0
    assert second["skipped_unchanged_files"] == 1
    assert pipeline.store.verify_consistency()["consistent"] is True

    # Modify file: old unit deleted, new unit indexed, no duplicates in Tantivy/Qdrant/SQLite
    py_file.write_text("def alpha(x: int) -> int:\n    return x + 99\n", encoding="utf-8")
    third = pipeline.index_workspace(repo_dir, incremental=True)
    assert third["indexed_units"] == 1
    assert third["deleted_units"] == 1
    consistency = pipeline.store.verify_consistency()
    assert consistency == {
        "consistent": True,
        "memory": 1,
        "tantivy": 1,
        "qdrant": 1,
        "sqlite": 1,
    }

    # Simulate process restart: close lock and recover from SQLite in a new store
    pipeline.store.close()
    restarted = HybridCodeStore(storage_dir=storage_dir)
    assert len(restarted.units) == 1
    assert restarted.verify_consistency()["consistent"] is True
    restarted.close()
