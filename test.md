# Verification & Test Specification (`/test` & `/verify`)

**Workspace Artifact:** `test.md`  
**Target Persona:** Antigravity Test Engineer (`test-engineer.md`)  
**Verification Pattern:** Prove-It Pattern (Deterministic Unit + Cross-Layer Equivalence + Performance SLA)

---

## 1. Verification Commands for Antigravity Terminal

The Antigravity agent must execute the following commands in the sandboxed terminal and confirm an exit code of `0` at each milestone:

```bash
# 1. Static Type & Lint Verification
ruff check src/ tests/
mypy --strict src/

# 2. Unit & Cross-Layer Golden Equivalence Suite
pytest tests/unit/ tests/golden/ -v --tb=short

# 3. Property-Based & Fuzz Testing (Hypothesis)
pytest tests/property/ -v --hypothesis-seed=42

# 4. End-to-End MCP Server & Performance Benchmark Suite
pytest tests/e2e/ tests/benchmarks/ -v
```

---

## 2. Golden Cross-Layer Equivalence Test Cases

The core invariant of the multi-layer indexer is that **semantically equivalent source constructs converge at the LLVM IR and Machine Code layers** while preserving bidirectional source-line traceability.

### Test Case G-01: Branchless Clamping Convergence (`tests/golden/test_clamp_convergence.py`)
Compile two syntactically distinct C implementations of a zero-clamping function at `-O1`:

* **Variant A (If-Else Statement):**
  ```c
  int clamp_branch(int x) {
      if (x > 0) return x;
      return 0;
  }
  ```
* **Variant B (Ternary Expression):**
  ```c
  int clamp_ternary(int x) {
      return (x > 0) ? x : 0;
  }
  ```

**Required Assertions:**
1. **Layer 1 (AST Divergence):** $\text{ASTHash}(\text{Variant A}) \neq \text{ASTHash}(\text{Variant B})$ (`if_statement` vs. `conditional_expression`).
2. **Layer 3 (LLVM IR Isomorphism):** Both functions lower to isomorphic ProGraML SSA graphs containing `icmp sgt i32 %x, 0` feeding `select i1 %cmp, i32 %x, i32 0`, yielding cosine similarity $S_{\text{cos}}(v_{\text{IR}}(A), v_{\text{IR}}(B)) \ge 0.99$.
3. **Layer 4 (Machine Code Identity):** Both functions disassemble to identical normalized `x86-64` instruction sequences:
   `["XOR REG32_A, REG32_A", "TEST REG32_ARG1, REG32_ARG1", "CMOVG REG32_A, REG32_ARG1", "RET"]`.

### Test Case G-02: Strength Reduction & Address Calculation (`tests/golden/test_strength_reduction.py`)
Verify that arithmetic scaling `x * 4` in `scale_pos(int x)` is indexed and linked across all four representations:
1. **AST Layer:** Captures `binary_expression` with operator `*` and literal `4`.
2. **LLVM IR Layer:** Captures lowered bit-shift instruction `shl nsw i32 %x, 2` with an explicit SSA data-dependency edge from parameter `%x`.
3. **Machine Code Layer:** Captures scaled-index instruction `lea edx, [rdi*4]` or `shl edi, 2` and maps its virtual address range back to the exact source line via DWARF `.debug_line`.

---

## 3. Test Matrix Across Pipeline Layers

| Test ID | Target Module | Test Scenario | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **UT-01** | `ast_chunker.py` | Parse file with syntax errors (incomplete function body) | Tree-sitter isolates `ERROR` node and still chunks sibling functions cleanly. |
| **UT-02** | `semantic_cpg.py` | Cross-file call resolution across `math.h` and `main.c` | Emits exact `CALLS` edge linking call site in `main.c` to definition URI in `math.c`. |
| **UT-03** | `llvm_ir_pass.py` | SSA Phi-node extraction on `while` loop with accumulator | Constructs cyclic CFG back-edge and `phi` data-flow edges with operand positions $p(e) \in \{0, 1\}$. |
| **UT-04** | `binary_lifter.py` | Address & immediate normalization under PIC relocation | Relocated offsets and stack displacements normalize to identical canonical token sequences. |
| **UT-05** | `dwarf_aligner.py` | Inlined function DWARF mapping (`DW_TAG_inlined_subroutine`) | Correctly attributes inlined machine-code basic blocks to both caller site and callee origin. |
| **PT-01** | `binary_lifter.py` | Property test: arbitrary random byte streams ($10^3$ samples) | Disassembler never crashes or hangs; returns graceful empty/invalid block metadata. |
| **E2E-01** | `code_index_server.py` | Antigravity MCP tool `inspect_lowering_pipeline("scale_pos")` | Returns complete JSON payload containing AST, SCIP symbol, LLVM IR, and normalized `x86-64` CFG. |

---

## 4. Quantitative Performance & Quality SLAs

All benchmark tests in `tests/benchmarks/test_indexing_sla.py` must satisfy the following bounds on a standard 4-vCPU Antigravity Linux sandbox:

1. **AST + Lexical Indexing Throughput:** $\ge 15,000\text{ LOC/s}$.
2. **Full 4-Layer Indexing Throughput (including Clang `-O1` IR + Binary Lifting):** $\ge 1,200\text{ LOC/s}$ using parallel worker pool.
3. **Incremental Single-File Re-index Latency:** $P_{95} \le 180\text{ ms}$ on file save.
4. **Hybrid RRF Query Latency (10k indexed functions):** $P_{95} \le 45\text{ ms}$, $P_{99} \le 85\text{ ms}$.
5. **Cross-Optimization Binary Clone Recall@5 (`-O0` vs. `-O2`):** $\ge 0.88$ on the golden benchmark fixture set.

---

## 5. Prove-It Verification Checklist (Agent Self-Check)

- [ ] Every test in `tests/unit/` and `tests/golden/` passes without mocking the Tree-sitter, LLVM, or Capstone engines.
- [ ] Fault injection test confirms that a broken C++ translation unit triggers the Layer 1 fallback without aborting the batch index run.
- [ ] MCP server responds to `tools/list` and `tools/call` over stdio within $50\text{ ms}$.