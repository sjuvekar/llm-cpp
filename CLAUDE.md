# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

This project uses **XMake** (not CMake). All build commands use the `xmake` CLI.

```bash
xmake build              # Build the main llm-cpp target
xmake build unit_tests   # Build unit tests
xmake run unit_tests     # Run all unit tests
xmake run llm-cpp        # Run the demo application
```

## Dependencies

XMake manages **libtorch** and **gtest** automatically via `xmake.lua`.

Two dependencies require manual installation:

**tiktoken-cpp** (BPE tokenizer):
```bash
git clone https://github.com/gh-markt/cpp-tiktoken thirdparty/cpp-tiktoken
cd thirdparty/cpp-tiktoken && mkdir build && cd build && cmake .. && make && sudo make install
```

**libomp** (macOS only):
```bash
brew install libomp
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib/:$DYLD_LIBRARY_PATH
```

## Architecture

The project implements LLM data pipeline components using C++26, PyTorch C++ API (libtorch), and tiktoken-compatible BPE tokenization.

**Core components** (all in `llm` namespace):

- **`ITokenizer`** (`include/tokenizer.h`) — Abstract interface with `encode(text, allowed_special) -> vector<int>`. The only concrete implementation is `TiktokenTokenizer` (`src/tokenizer.cpp`), which wraps `cpp-tiktoken`'s `GptEncoding`. Tests use `MockTokenizer` (`tests/mocks/mock_tokenizer.h`) via GoogleMock.

- **`GPTDataset`** (`include/dataset.h`, `src/dataset.cpp`) — Extends `torch::data::Dataset<GPTDataset>`. Tokenizes text once at construction, then creates overlapping sliding windows of `(input_ids, target_ids)` pairs for next-token prediction. `target` is `input` shifted by one position. Sample count: `(token_count - window_length) / stride`.

- **`DataLoaderConfig` + `create_dataloader()`** (`include/dataloader.h`) — Header-only factory that composes `GPTDataset` + torch's `DataLoader` with batching, shuffling, and drop-last semantics.

**Key design decisions:**
- `GPTDataset` accepts `shared_ptr<ITokenizer>`, enabling injection of mock tokenizers in tests.
- Tokenizer files in `tokenizers/` are loaded at runtime by `TiktokenTokenizer` (r50k_base, cl100k_base, o200k_base, qwen, llama3.1, etc.).
- GPU support is disabled by default in `xmake.lua` (`cuda = false`).

## Testing

Tests use **Google Test + Google Mock**. Test file: `tests/dataset_test.cpp`. The `MockTokenizer` in `tests/mocks/mock_tokenizer.h` implements `ITokenizer` using `MOCK_METHOD`.

```bash
xmake build unit_tests && xmake run unit_tests
```

To run a specific test (using GTest filter):
```bash
xmake run unit_tests -- --gtest_filter=GPTDatasetTest.BasicDatasetCreation
```
