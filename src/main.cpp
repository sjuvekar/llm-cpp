#include "dataloader.h"
#include "embedding.h"

#include <iostream>

int main() {
    // GPT-2 small hyperparameters
    constexpr int64_t vocab_size     = 50257; // r50k_base vocabulary
    constexpr int64_t context_length = 256;   // positional table size
    constexpr int64_t emb_dim        = 768;   // GPT-2 small hidden size
    constexpr int64_t batch_size     = 2;
    constexpr int64_t max_length     = 16;    // tokens per sample
    constexpr int64_t stride         = 16;    // no overlap between samples

    // Opening of "The Verdict" by Edith Wharton — the sample text used in
    // Raschka's "Build a Large Language Model (From Scratch)".
    const std::string text =
        "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no "
        "great surprise to me to hear that, in the height of his glory, he had dropped his painting, married "
        "a widow, and established himself in a ghastly Normandy chateau that would make most honest men "
        "shudder. The news reached me at a moment when I was feeling a sudden disgust for my own work, and "
        "it was perhaps this that made me more ready than usual to let myself be drawn into a somewhat "
        "intimate correspondence with Jack. His letters were full of the chateau--the towers, the moat, the "
        "chapel--and of the life he led there--the shooting, the hunting, the long evenings by the library "
        "fire. It all sounded uncommonly pleasant, and it was only gradually that I began to notice a "
        "certain reticence about the painting itself.";

    // -------------------------------------------------------------------------
    // Data pipeline
    // -------------------------------------------------------------------------
    llm::DataLoaderConfig config;
    config.language_model = LanguageModel::R50K_BASE;
    config.batch_size  = static_cast<unsigned long>(batch_size);
    config.max_length  = static_cast<unsigned long>(max_length);
    config.stride      = static_cast<unsigned long>(stride);
    config.shuffle     = false;
    config.drop_last   = true;
    config.num_workers = 0;

    auto dataloader = llm::create_dataloader(text, config);

    // -------------------------------------------------------------------------
    // Embedding layer
    // -------------------------------------------------------------------------
    auto embedding = llm::EmbeddingLayer(vocab_size, context_length, emb_dim);
    embedding.eval(); // inference mode — disables dropout if added later

    std::cout << "EmbeddingLayer parameters: "
              << (vocab_size + context_length) * emb_dim << " floats\n"
              << "  token_emb : [" << vocab_size     << " x " << emb_dim << "]\n"
              << "  pos_emb   : [" << context_length << " x " << emb_dim << "]\n\n";

    // -------------------------------------------------------------------------
    // One forward pass through the first batch
    // -------------------------------------------------------------------------
    for (auto& batch : *dataloader) {
        auto token_ids = batch.data;    // [batch_size, max_length]  int64
        auto targets   = batch.target;  // [batch_size, max_length]  int64

        std::cout << "Input token IDs  : " << token_ids.sizes() << "\n";
        std::cout << "Target token IDs : " << targets.sizes()   << "\n";
        std::cout << "First sample     : " << token_ids[0]      << "\n\n";

        torch::NoGradGuard no_grad;
        auto embeddings = embedding.forward(token_ids); // [batch_size, max_length, emb_dim]

        std::cout << "Embeddings shape : " << embeddings.sizes() << "\n";
        std::cout << "Embeddings dtype : " << embeddings.dtype() << "\n";
        std::cout << "First token vec  : " << embeddings[0][0].slice(0, 0, 8) << " ...\n";

        break; // one batch is enough for the demo
    }
}
