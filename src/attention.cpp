/**
 * @file attention.cpp
 * @brief CausalSelfAttention implementation
 *
 * @author Sudeep Juvekar (sjuvekar@gmail.com)
 */

#include "attention.h"

#include <cmath>
#include <stdexcept>

namespace llm {

CausalSelfAttention::CausalSelfAttention(
    int64_t embedding_dim,
    int64_t num_heads,
    int64_t context_length,
    double  dropout,
    bool    qkv_bias)
    : num_heads_(num_heads),
      head_dim_(embedding_dim / num_heads)
{
    if (embedding_dim % num_heads != 0) {
        throw std::invalid_argument(
            "embedding_dim must be divisible by num_heads");
    }

    torch::nn::LinearOptions proj_opts(embedding_dim, embedding_dim);
    proj_opts.bias(qkv_bias);

    W_query  = register_module("W_query",  torch::nn::Linear(proj_opts));
    W_key    = register_module("W_key",    torch::nn::Linear(proj_opts));
    W_value  = register_module("W_value",  torch::nn::Linear(proj_opts));
    out_proj = register_module("out_proj", torch::nn::Linear(
        torch::nn::LinearOptions(embedding_dim, embedding_dim).bias(true)));

    attn_drop  = register_module("attn_drop",  torch::nn::Dropout(dropout));
    resid_drop = register_module("resid_drop", torch::nn::Dropout(dropout));

    // Causal mask: upper-triangular matrix (above diagonal = 1)
    // Registered as a non-parameter buffer so it moves with .to(device).
    auto mask = torch::triu(
        torch::ones({context_length, context_length}, torch::kBool), /*diagonal=*/1);
    register_buffer("mask", mask);
}

torch::Tensor CausalSelfAttention::forward(torch::Tensor x) {
    // x: [batch, seq_len, embedding_dim]
    const int64_t batch   = x.size(0);
    const int64_t seq_len = x.size(1);

    // Project to Q, K, V — each [batch, seq_len, embedding_dim]
    auto q = W_query->forward(x);
    auto k = W_key->forward(x);
    auto v = W_value->forward(x);

    // Reshape to [batch, seq_len, num_heads, head_dim]
    // then transpose to [batch, num_heads, seq_len, head_dim]
    auto reshape = [&](torch::Tensor t) {
        return t.view({batch, seq_len, num_heads_, head_dim_})
                .transpose(1, 2)
                .contiguous();
    };
    q = reshape(q);
    k = reshape(k);
    v = reshape(v);

    // Scaled dot-product attention scores: [batch, num_heads, seq_len, seq_len]
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim_));
    auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;

    // Apply causal mask: mask out future positions with -inf
    auto causal_mask = named_buffers()["mask"]
                           .slice(0, 0, seq_len)
                           .slice(1, 0, seq_len);
    scores = scores.masked_fill(causal_mask, -std::numeric_limits<float>::infinity());

    // Softmax over the key dimension, then dropout
    auto attn_weights = torch::softmax(scores, /*dim=*/-1);
    attn_weights = attn_drop->forward(attn_weights);

    // Weighted sum of values: [batch, num_heads, seq_len, head_dim]
    auto context = torch::matmul(attn_weights, v);

    // Merge heads: [batch, seq_len, embedding_dim]
    context = context.transpose(1, 2).contiguous()
                     .view({batch, seq_len, num_heads_ * head_dim_});

    // Output projection + residual dropout
    return resid_drop->forward(out_proj->forward(context));
}

} // namespace llm
