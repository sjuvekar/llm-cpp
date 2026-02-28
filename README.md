# llm-cpp

A C++ implementation of Large Language Model components, inspired by "Build a Large Language Model (From Scratch)" by Sebastian Raschka.

This project provides high-performance C++ implementations of LLM building blocks, including tokenization, dataset creation, and data loading pipelines.

## Third-Party Libraries

This project uses the following third-party libraries:

- **[tiktoken-cpp](https://github.com/gh-markt/cpp-tiktoken)** - BPE tokenizer compatible with GPT-2/GPT-4 models
- **libtorch** - PyTorch C++ API for tensor operations

## Requirements

- **xmake** - Build system
- **libomp** - OpenMP support
- **tiktoken-cpp** - BPE tokenizer library

## Installation

### 1. Install OpenMP (macOS)

```bash
brew install libomp
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib/:$DYLD_LIBRARY_PATH
```

### 2. Install tiktoken-cpp

```bash
# Clone the library
git clone https://github.com/gh-markt/cpp-tiktoken thirdparty/cpp-tiktoken

# Build and install
cd thirdparty/cpp-tiktoken
mkdir build && cd build
cmake ..
make
sudo make install
```

### 3. Install xmake

Follow the instructions at [xmake.io](https://xmake.io/#/guide/installation) for your platform.

## Build

```bash
xmake build
```

## Project Structure

```
llm-cpp/
├── include/          # Header files
│   ├── dataloader.h  # DataLoader implementation
│   └── dataset.h     # GPTDataset class
├── src/              # Source files
│   ├── main.cpp      # Entry point
│   └── dataset.cpp   # Dataset implementation
├── tokenizers/       # Pre-trained tokenizer files
├── tests/            # Unit tests
└── xmake.lua         # Build configuration
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
