/**
 * @file attention.h
 * @brief Causal multi-head self-attention for GPT-style models
 *
 * @author Sudeep Juvekar (sjuvekar@gmail.com)
 */

#pragma once

#include <torch/torch.h>

namespace llm {

/**
 * @brief Causal multi-head self-attention layer.
 *
 * Implements scaled dot-product self-attention with a causal (autoregressive)
 * mask so each position can only attend to earlier positions and itself.
 *
 * The input projection is split into Q, K, V heads, attention scores are
 * computed and masked, then the weighted values are concatenated and passed
 * through an output projection.
 *
 * Example:
 * @code
 * auto attn = llm::CausalSelfAttention(768, 12, 1024, 0.1);
 * torch::Tensor x = torch::randn({2, 16, 768});  // [batch, seq, emb_dim]
 * torch::Tensor y = attn->forward(x);            // [2, 16, 768]
 * @endcode
 */
struct CausalSelfAttention : torch::nn::Module {
    /**
     * @param embedding_dim   Dimensionality of the input and output (d_model).
     *                        Must be divisible by num_heads.
     * @param num_heads       Number of parallel attention heads.
     * @param context_length  Maximum sequence length; used to pre-build the
     *                        causal mask buffer.
     * @param dropout         Dropout probability applied to attention weights
     *                        and residual output (0.0 disables dropout).
     * @param qkv_bias        Whether to include bias terms in Q/K/V projections.
     */
    CausalSelfAttention(
        int64_t embedding_dim,
        int64_t num_heads,
        int64_t context_length,
        double  dropout  = 0.0,
        bool    qkv_bias = false);

    /**
     * @brief Run causal self-attention over the input sequence.
     *
     * @param x Float tensor of shape [batch, seq_len, embedding_dim].
     *          seq_len must not exceed context_length.
     * @return  Float tensor of the same shape [batch, seq_len, embedding_dim].
     */
    torch::Tensor forward(torch::Tensor x);

    int64_t num_heads_;
    int64_t head_dim_;

    torch::nn::Linear W_query{nullptr};
    torch::nn::Linear W_key{nullptr};
    torch::nn::Linear W_value{nullptr};
    torch::nn::Linear out_proj{nullptr};

    torch::nn::Dropout attn_drop{nullptr};
    torch::nn::Dropout resid_drop{nullptr};
};

} // namespace llm
