#include "dataset.h"
#include "tokenizer.h"

#include <tiktoken/encoding.h>
#include <iostream>

int main() {
    // Create the tokenizer
    auto tokenizer = llm::create_tiktoken_tokenizer(LanguageModel::R50K_BASE);

    // Create the dataset with the tokenizer
    auto gpt_dataset = llm::GPTDataset(
        "hello world. This is a test of long sentence. A mouse took a stroll through a deep dark wood.",
        /*window_length=*/3,
        /*stride=*/2,
        tokenizer
    );
    std::cout << "Dataset size: " << *gpt_dataset.size() << std::endl;
}