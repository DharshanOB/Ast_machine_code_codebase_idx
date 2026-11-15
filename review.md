# Code & Architecture Review Artifact (`/review`)

**Workspace Artifact:** `review.md`  
**Target Personas:** Antigravity Code Reviewer (`code-reviewer.md`) & Security Auditor (`security-auditor.md`)  
**Review Scope:** Multi-Layer AI Code Indexer (`AST` $\rightarrow$ `SCIP/CPG` $\rightarrow$ `LLVM IR` $\rightarrow$ `Machine Code`)

---

## 1. Five-Axis Engineering Rubric

| Axis | Weight | Score ($1\text{--}5$) | Status | Key Evaluation Criteria |
| :--- | :---: | :---: | :---: | :--- |
| **1. Architectural Soundness** | $25\%$ | $4.8 / 5.0$ | PASS | Clean separation between parsing, compiler lowering, DWARF alignment, and tri-modal storage; graceful degradation when builds fail. |
| **2. Sandbox & Compiler Security** | $25\%$ | $4.6 / 5.0$ | PASS (with guards) | Strict subprocess argument isolation for `clang-18`; prevention of macro-expansion bombs and malicious `compile_commands.json` flags. |
| **3. Performance & Memory Safety** | $20\%$ | $4.7 / 5.0$ | PASS | Bounded streaming for large `.ll` and ELF files; zero unbounded graph traversals on cyclic CFGs. |
| **4. Schema & Cross-Layer Determinism** | $15\%$ | $4.9 / 5.0$ | PASS | Content-addressable SHA-256 IDs; reproducible SSA register and assembly operand normalization across runs. |
| **5. Antigravity MCP Ergonomics** | $15\%$ | $4.8 / 5.0$ | PASS | Concise, token-efficient MCP responses with pagination and layer filtering to prevent agent context window bloat. |

**Composite Readiness Score:** $4.75 / 5.00$ (**Approved for Execution with Mandatory Security Guards**)

---

## 2. Security & Hardening Audit (`security-auditor.md`)

Indexing untrusted repositories inside an automated coding agent introduces compiler and parser attack surfaces that must be hardened prior to `/ship`:

### Finding SEC-01 `[BLOCKER]`: Arbitrary Flag Injection via `compile_commands.json`
* **Risk:** A malicious repository can include compiler flags such as `-fplugin=/path/to/malicious.so` or `@response_file` inside `compile_commands.json`, executing arbitrary code when the indexer invokes `clang-18` to emit LLVM IR.
* **Required Remediation:** Implement a strict flag allowlist filter in `src/indexer/llvm_ir_pass.py` that strips all flags except safe include paths (`-I`, `-isystem`), macro definitions (`-D`, `-U`), and language standard flags (`-std=`), and always invokes `clang-18` via `subprocess.run(..., shell=False)`.

```python
# Required Sanitizer in src/indexer/llvm_ir_pass.py
ALLOWED_CLANG_PREFIXES: tuple[str, ...] = ("-I", "-isystem", "-D", "-U", "-std=", "-target ")

def sanitize_compile_flags(raw_flags: list[str]) -> list[str]:
    safe_flags: list[str] = []
    skip_next = False
    for i, flag in enumerate(raw_flags):
        if skip_next:
            skip_next = False
            continue
        if flag in ("-I", "-isystem", "-D", "-U") and i + 1 < len(raw_flags):
            safe_flags.extend([flag, raw_flags[i + 1]])
            skip_next = True
        elif flag.startswith(ALLOWED_CLANG_PREFIXES) and "-fplugin" not in flag:
            safe_flags.append(flag)
    return safe_flags
```

### Finding SEC-02 `[WARNING]`: Exponential Macro Expansion & Template Instantiation Bombs
* **Risk:** Pathological C++ templates or deeply nested C macros can cause `clang -emit-llvm` or `llvmlite` IR graph construction to consume multiple gigabytes of RAM.
* **Required Remediation:**
  1. Pass `-ferror-limit=20 -ftemplate-depth=64` to Clang during IR extraction.
  2. Enforce `resource.RLIMIT_AS` ($2\text{ GB}$ virtual memory cap) and a hard timeout of $T_{\text{max}} = 15\text{ s}$ on every compiler subprocess.
  3. Cap maximum nodes per function graph at $|V_{\text{max}}| = 10,000$; truncate and flag oversized generated functions (e.g., protobuf tables).

### Finding SEC-03 `[WARNING]`: Agent Context Window Exhaustion on Large IR/Assembly Dumps
* **Risk:** Returning full LLVM IR modules or disassembled `.text` sections through MCP tool calls can rapidly trigger Antigravity's $135\text{k}$ token context compaction threshold.
* **Required Remediation:** Enforce a default token budget of $2,048\text{ tokens}$ per MCP tool response in `src/mcp/code_index_server.py`, returning basic-block summaries and offering cursor-based pagination by Basic Block ID (`bb_cursor`).

---

## 3. Code Quality & Architectural Review Notes

* **`[WARNING]` Cyclic Control Flow Graph Traversal (`src/indexer/semantic_cpg.py`):**
  When computing $k$-hop backward data-flow slices over loops with phi nodes, ensure the BFS visitor tracks `(node_id, hop_depth)` in a visited set so back-edges do not cause redundant exponential path enumeration.
* **`[NIT]` Deterministic Dictionary Ordering in Graph Hashing (`src/storage/hybrid_store.py`):**
  When computing Weisfeiler-Lehman subtree hashes over ProGraML IR graphs, sort neighbor labels lexicographically before hashing so that multi-threaded LLVM pass emission order never alters the structural embedding vector.
* **`[NIT]` DWARF Column Zero Handling (`src/indexer/dwarf_aligner.py`):**
  Compiler-generated prologue and epilogue instructions often carry DWARF line `0` or column `0`. Map line `0` instructions to the enclosing function's entry span rather than dropping them from the binary CFG index.

---

## 4. Pre-Ship Sign-Off Checklist (`/ship` Gate)

- [x] `research.md` architectural invariants and 4-layer representations validated.
- [x] `implementation_plan.md` module boundaries and MCP tool signatures finalized.
- [x] `SEC-01` (`compile_commands.json` flag sanitizer) and `SEC-02` (subprocess memory/time limits) integrated into Phase 2 implementation specs.
- [ ] All automated checks in `test.md` executed with zero failures in the Antigravity terminal.