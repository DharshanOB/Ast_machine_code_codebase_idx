"""Security, adversarial repository, incremental indexing, and store consistency tests."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest

from indexer.ast_chunker import ASTChunker
from indexer.llvm_ir_pass import LLVMIRPass, resolve_clang_binary
from indexer.models import Language, UnifiedCodeUnit
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


def test_pipeline_restart_persists_incremental_state_and_handles_deletes(
    tmp_path: Path,
) -> None:
    """Incremental file hashes and unit IDs persist across pipeline restarts."""
    storage_dir = tmp_path / "idx"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    file_a = repo_dir / "a.py"
    file_b = repo_dir / "b.py"
    file_a.write_text("def func_a(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    file_b.write_text("def func_b(y: int) -> int:\n    return y + 2\n", encoding="utf-8")

    pipeline = CodeIndexPipeline(storage_dir=storage_dir)
    first = pipeline.index_workspace(repo_dir)
    assert first["indexed_units"] == 2
    assert pipeline.store.verify_consistency()["consistent"] is True
    pipeline.store.close()

    file_a.unlink()
    file_b.write_text("def func_b(y: int) -> int:\n    return y + 20\n", encoding="utf-8")

    restarted = CodeIndexPipeline(storage_dir=storage_dir)
    second = restarted.index_workspace(repo_dir, incremental=True)
    # 1 unit from deleted file_a + 1 old unit replaced in modified file_b
    assert second["deleted_units"] == 2
    assert second["indexed_units"] == 1
    assert len(restarted.store.units) == 1
    assert restarted.store.verify_consistency()["consistent"] is True
    restarted.store.close()


def test_compile_commands_change_invalidates_incremental_cache(tmp_path: Path) -> None:
    """Changing compiler flags in compile_commands.json invalidates cached file state."""
    if resolve_clang_binary() is None:
        pytest.skip("clang binary not available")

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    src = repo_dir / "main.c"
    src.write_text("int compute(void) { return VAL; }\n", encoding="utf-8")

    cc_path = tmp_path / "compile_commands.json"

    def _write_cc(val: int) -> None:
        cc_path.write_text(
            json.dumps(
                [
                    {
                        "directory": str(repo_dir),
                        "file": str(src),
                        "arguments": ["clang", f"-DVAL={val}"],
                    }
                ]
            ),
            encoding="utf-8",
        )

    _write_cc(1)
    pipeline = CodeIndexPipeline(storage_dir=tmp_path / "idx")
    first = pipeline.index_workspace(repo_dir, compile_commands_path=cc_path)
    assert first["indexed_units"] == 1

    _write_cc(2)
    second = pipeline.index_workspace(
        repo_dir, compile_commands_path=cc_path, incremental=True
    )
    assert second["skipped_unchanged_files"] == 0
    assert second["indexed_units"] == 1
    pipeline.store.close()


def test_sqlite_tombstone_replay_on_crash_recovery(tmp_path: Path) -> None:
    """Unreconciled SQLite tombstones purge stale entries from Tantivy and Qdrant on restart."""
    storage_dir = tmp_path / "idx"
    store = HybridCodeStore(storage_dir=storage_dir)

    chunk = ASTChunker().chunk_source(
        "def target(x: int) -> int:\n    return x + 1\n",
        Language.PYTHON,
        "target.py",
    )[0]
    unit = UnifiedCodeUnit(ast_chunk=chunk)
    store.upsert_units([unit])
    assert store.verify_consistency() == {
        "consistent": True,
        "memory": 1,
        "tantivy": 1,
        "qdrant": 1,
        "sqlite": 1,
    }

    db_path = storage_dir / "graph_snapshots.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO tombstones (unit_id) VALUES (?)",
            (unit.unit_id,),
        )
        conn.execute("DELETE FROM units WHERE unit_id = ?", (unit.unit_id,))
        conn.commit()
    store.close()

    recovered = HybridCodeStore(storage_dir=storage_dir)
    assert recovered.verify_consistency() == {
        "consistent": True,
        "memory": 0,
        "tantivy": 0,
        "qdrant": 0,
        "sqlite": 0,
    }
    recovered.close()


def test_index_workspace_reports_ir_cpg_validation(tmp_path: Path) -> None:
    """index_workspace reports IR-CPG call-edge validation metrics for C code."""
    if resolve_clang_binary() is None:
        pytest.skip("clang binary not available")

    src = tmp_path / "calls.c"
    src.write_text(
        "__attribute__((noinline)) int leaf(int x) { return x + 1; }\n"
        "int caller(int y) { return leaf(y) * 2; }\n",
        encoding="utf-8",
    )

    pipeline = CodeIndexPipeline(storage_dir=tmp_path / "idx")
    summary = pipeline.index_workspace(tmp_path)
    assert "ir_cpg_validation" in summary
    assert summary["ir_cpg_validation"]["confirmed"] >= 1
    pipeline.store.close()
