/**
 * @file embedding.cpp
 * @brief EmbeddingLayer implementation
 *
 * @author Sudeep Juvekar (sjuvekar@gmail.com)
 */

#include "embedding.h"

namespace llm {

EmbeddingLayer::EmbeddingLayer(
    int64_t vocab_size, int64_t context_length, int64_t embedding_dim)
    : token_emb(register_module(
          "token_emb", torch::nn::Embedding(vocab_size, embedding_dim))),
      pos_emb(register_module(
          "pos_emb", torch::nn::Embedding(context_length, embedding_dim))) {}

torch::Tensor EmbeddingLayer::forward(torch::Tensor token_ids) {
    int64_t seq_len = token_ids.size(-1);

    // Position indices [0, 1, ..., seq_len-1] on the same device as token_ids
    auto pos_ids = torch::arange(seq_len, torch::kLong).to(token_ids.device());

    // token_emb: [..., seq_len, emb_dim]
    // pos_emb:   [seq_len, emb_dim]  — broadcasts over the batch dimension
    return token_emb->forward(token_ids) + pos_emb->forward(pos_ids);
}

} // namespace llm
