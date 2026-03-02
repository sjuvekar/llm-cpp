/**
 * @file embedding.h
 * @brief Token and positional embedding layer for GPT-style models
 *
 * @author Sudeep Juvekar (sjuvekar@gmail.com)
 */

#pragma once

#include <torch/torch.h>

namespace llm {

/**
 * @brief Combined token + positional embedding layer.
 *
 * Maintains two learned embedding tables:
 * - token_emb: maps each token ID to a dense vector (vocab_size x embedding_dim)
 * - pos_emb:   maps each position index to a dense vector (context_length x embedding_dim)
 *
 * The output is the element-wise sum of the two embeddings, which is the
 * standard input representation for GPT-style transformer models.
 *
 * Example:
 * @code
 * auto emb = llm::EmbeddingLayer(50257, 1024, 768);  // GPT-2 dims
 * torch::Tensor token_ids = torch::randint(50257, {2, 16});  // [batch=2, seq=16]
 * torch::Tensor x = emb->forward(token_ids);                 // [2, 16, 768]
 * @endcode
 */
struct EmbeddingLayer : torch::nn::Module {
    /**
     * @param vocab_size      Number of unique tokens in the vocabulary
     * @param context_length  Maximum sequence length (sets positional table size)
     * @param embedding_dim   Dimensionality of each embedding vector
     */
    EmbeddingLayer(int64_t vocab_size, int64_t context_length, int64_t embedding_dim);

    /**
     * @brief Embed token IDs into continuous vectors.
     *
     * @param token_ids Integer tensor of shape [seq_len] or [batch, seq_len].
     *                  Values must be in [0, vocab_size) and seq_len must not
     *                  exceed context_length.
     * @return Float tensor of shape [seq_len, emb_dim] or [batch, seq_len, emb_dim].
     */
    torch::Tensor forward(torch::Tensor token_ids);

    torch::nn::Embedding token_emb{nullptr};
    torch::nn::Embedding pos_emb{nullptr};
};

} // namespace llm
