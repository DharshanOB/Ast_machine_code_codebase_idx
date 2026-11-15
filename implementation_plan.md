# Implementation Plan: Multi-Layer AI Code Indexer (AST to Machine Code)

**Artifact Type:** Antigravity Implementation Plan (`/plan`)  
**Status:** Ready for Human Review (Click **Proceed** or attach inline comments to adjust scope)  
**Target Workspace:** `antigravity-code-indexer/`

---

## 1. Objective & Architectural Summary

Build a modular, production-grade code indexing pipeline and Antigravity MCP server that ingests source repositories, compiles translation units to LLVM SSA IR and target machine code, aligns all four representation layers via DWARF debug metadata, and exposes hybrid retrieval tools to Antigravity coding agents.

```
Repository Source ──► [Layer 1: Tree-sitter AST Chunker] ──┐
                  ──► [Layer 2: SCIP & CPG Linker]       ──┼──► [DWARF / Span Aligner] ──► [Tri-Modal Store] ──► [Antigravity MCP Server]
compile_commands  ──► [Layer 3: LLVM SSA IR Extractor]   ──┤
ELF / Object Bin  ──► [Layer 4: Capstone Binary Lifter]  ──┘
```

---

## 2. Antigravity Execution & Sandbox Policy

* **Terminal Execution Policy:** Sandbox-isolated execution enabled. Compiler invocations (`clang-18`) must run with `-fsyntax-only` or emit to an isolated temporary build directory `/tmp/ag-index-build/` with a per-file timeout of $T_{\text{max}} = 15\text{ s}$.
* **Network Policy:** Local-only execution during indexing (embedded Qdrant, Tantivy, and SQLite/NetworkX graph store); zero external egress required.
* **Artifact Checkpoints:** Agent pauses after Phase 2 (IR/Binary extraction) and Phase 4 (MCP Server integration) if verification tests fail.

---

## 3. Proposed File Changes

### Component 1: Core Schema & Cross-Layer Alignment
* **`[NEW]` `src/indexer/models.py` (~180 lines)**
  * Define Pydantic v2 schemas for `UnifiedCodeUnit`, `ASTChunkNode`, `SCIPSymbolNode`, `LLVMIRGraph`, `BinaryBasicBlock`, and `DWARFLineMapping`.
  * Compute deterministic content-addressable identifiers using SHA-256 over normalized representations.
* **`[NEW]` `src/indexer/dwarf_aligner.py` (~210 lines)**
  * Parse `.debug_line` tables from compiled object files using `pyelftools` / `lief`.
  * Construct an interval tree mapping source line ranges $[l_{\text{start}}, l_{\text{end}}]$ to LLVM IR `!dbg` metadata IDs and machine code virtual address intervals $[\text{VA}_{\text{start}}, \text{VA}_{\text{end}})$.

### Component 2: Layer 1 & Layer 2 (AST & Semantic Graph)
* **`[NEW]` `src/indexer/ast_chunker.py` (~240 lines)**
  * Initialize `tree-sitter` parsers for C, C++, Rust, and Python.
  * Implement structure-aware cAST chunking with parent-scope context headers and docstring binding.
* **`[NEW]` `src/indexer/semantic_cpg.py` (~260 lines)**
  * Ingest `index.scip` protobuf payloads or build intraprocedural CFG/DFG edges over AST nodes.
  * Resolve cross-file call edges (`CALLS`), type inheritance (`IMPLEMENTS`), and variable def-use chains (`REACHES`).

### Component 3: Layer 3 (LLVM SSA IR Extractor)
* **`[NEW]` `src/indexer/llvm_ir_pass.py` (~310 lines)**
  * Invoke `clang-18 -S -emit-llvm -O1 -gline-tables-only -fno-discard-value-names` on target source files.
  * Parse emitted `.ll` modules using `llvmlite.binding` to extract basic blocks, SSA instructions, phi nodes, and typed data-dependency edges in ProGraML format.
  * Canonicalize local SSA register identifiers (`%0`, `%1`, `%cmp` $\rightarrow$ `<SSA_REG_i>`) while preserving instruction opcodes (`icmp sgt`, `shl nsw`, `select`).

### Component 4: Layer 4 (Machine Code Lifter & Normalizer)
* **`[NEW]` `src/indexer/binary_lifter.py` (~290 lines)**
  * Compile translation units to relocatable object files (`.o`) and parse `.text` symbols with `lief`.
  * Disassemble function byte ranges using `capstone` (`CS_ARCH_X86`, `CS_MODE_64` and `CS_ARCH_ARM64`).
  * Reconstruct binary Basic Block Control Flow Graphs (BB-CFG) by splitting on conditional/unconditional jumps (`jcc`, `jmp`, `ret`) and resolving jump target edges.
  * Apply instruction normalization $\Phi(I)$ for PalmTree/jTrans-compatible token sequences.

### Component 5: Tri-Modal Hybrid Storage & Query Router
* **`[NEW]` `src/storage/hybrid_store.py` (~340 lines)**
  * Provision embedded `Tantivy` lexical/trigram index, local `Qdrant` vector collection (multi-vector per unit: `ast_dense`, `ir_structural`, `bin_cfg`), and persistent property graph.
  * Implement weighted Reciprocal Rank Fusion (`rrf_search`) and multi-hop graph neighborhood expansion.

### Component 6: Antigravity MCP Server & CLI Entrypoint
* **`[NEW]` `src/mcp/code_index_server.py` (~220 lines)**
  * Register FastMCP tools for the Antigravity agent:
    * `index_workspace(repo_path: str, compile_commands_path: str | None)`
    * `search_code_hybrid(query: str, layers: list[str], top_k: int)`
    * `inspect_lowering_pipeline(symbol_or_function: str)`
    * `find_ir_or_binary_clones(function_id: str, threshold: float)`
    * `trace_symbol_dataflow(symbol_id: str, max_hops: int)`
* **`[MODIFY]` `.antigravity/mcp_servers.json` (~25 lines)**
  * Register the `code-multi-indexer` local stdio MCP server so Antigravity IDE and CLI automatically discover the indexing tools.

---

## 4. Phased Execution Milestones

### Phase 1: Foundation & AST/SCIP Indexing
1. Create project structure, `pyproject.toml`, and `src/indexer/models.py`.
2. Implement `src/indexer/ast_chunker.py` and `src/indexer/semantic_cpg.py`.
3. **Checkpoint 1:** Run `pytest tests/test_ast_and_cpg.py` to verify AST boundary chunking and cross-file symbol linking.

### Phase 2: Compiler IR & Machine Code Pipeline
1. Implement `src/indexer/llvm_ir_pass.py` to compile C/C++/Rust fixtures into SSA IR graphs.
2. Implement `src/indexer/binary_lifter.py` and `src/indexer/dwarf_aligner.py` to extract normalized `x86-64` basic-block CFGs and align them via DWARF line tables.
3. **Checkpoint 2:** Run `pytest tests/test_ir_and_binary.py` to verify that source functions map deterministically to their lowered `select`/`cmovg` IR and machine code blocks.

### Phase 3: Tri-Modal Index & RRF Retrieval
1. Implement `src/storage/hybrid_store.py` integrating Tantivy, Qdrant, and Graph storage.
2. Implement structural graph feature hashing (Weisfeiler-Lehman subtree kernel + token embeddings) for fast local IR/CFG vectorization.
3. **Checkpoint 3:** Run `pytest tests/test_hybrid_retrieval.py` to validate cross-layer clone detection and RRF ranking.

### Phase 4: Antigravity MCP Integration & End-to-End Verification
1. Implement `src/mcp/code_index_server.py` and wire `.antigravity/mcp_servers.json`.
2. Execute the full verification suite defined in `test.md`.

---

## 5. Rollback & Failure Recovery

* **Uncompilable Source Files:** If a file fails Clang/LLVM compilation (missing headers or syntax errors during live editing), the indexer gracefully degrades to **Layer 1 (Tree-sitter error-tolerant AST)** and **Layer 2 (heuristic lexical symbol graph)** while marking `ir_status = "DEGRADED"` in the index record.
* **Index Corruption Recovery:** All storage writes occur inside atomic batch transactions keyed by Git commit SHA and file content hash, enabling instant rollback to the last verified snapshot.