/**
 * @file embedding_test.cpp
 * @brief Unit tests for EmbeddingLayer
 *
 * @author Sudeep Juvekar (sjuvekar@gmail.com)
 */

#include <gtest/gtest.h>
#include "embedding.h"

namespace llm {
namespace testing {

// Shared dims used across tests
static constexpr int64_t kVocabSize      = 100;
static constexpr int64_t kContextLength  = 32;
static constexpr int64_t kEmbeddingDim   = 16;

class EmbeddingLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        emb_ = std::make_unique<EmbeddingLayer>(kVocabSize, kContextLength, kEmbeddingDim);
    }

    std::unique_ptr<EmbeddingLayer> emb_;
};

// ============================================================================
// Output shape tests
// ============================================================================

TEST_F(EmbeddingLayerTest, Output1DShape) {
    // Input: [seq_len]  ->  Output: [seq_len, emb_dim]
    int64_t seq_len = 8;
    auto token_ids = torch::randint(kVocabSize, {seq_len}, torch::kLong);
    auto out = emb_->forward(token_ids);

    EXPECT_EQ(out.dim(), 2);
    EXPECT_EQ(out.size(0), seq_len);
    EXPECT_EQ(out.size(1), kEmbeddingDim);
}

TEST_F(EmbeddingLayerTest, Output2DShape) {
    // Input: [batch, seq_len]  ->  Output: [batch, seq_len, emb_dim]
    int64_t batch_size = 4;
    int64_t seq_len    = 8;
    auto token_ids = torch::randint(kVocabSize, {batch_size, seq_len}, torch::kLong);
    auto out = emb_->forward(token_ids);

    EXPECT_EQ(out.dim(), 3);
    EXPECT_EQ(out.size(0), batch_size);
    EXPECT_EQ(out.size(1), seq_len);
    EXPECT_EQ(out.size(2), kEmbeddingDim);
}

// ============================================================================
// Output dtype test
// ============================================================================

TEST_F(EmbeddingLayerTest, OutputDType) {
    auto token_ids = torch::randint(kVocabSize, {4}, torch::kLong);
    auto out = emb_->forward(token_ids);
    // nn::Embedding weights default to float32
    EXPECT_EQ(out.dtype(), torch::kFloat32);
}

// ============================================================================
// Positional encoding tests
// ============================================================================

TEST_F(EmbeddingLayerTest, DifferentPositionsProduceDifferentEmbeddings) {
    // Same token ID at different positions should produce different embeddings
    // because the positional embedding differs per position.
    int64_t seq_len = 4;
    auto token_ids = torch::zeros({seq_len}, torch::kLong);  // all same token
    auto out = emb_->forward(token_ids);  // [seq_len, emb_dim]

    // Rows at position 0 and 1 differ (pos_emb adds different offsets)
    EXPECT_FALSE(out[0].equal(out[1]));
}

TEST_F(EmbeddingLayerTest, SameInputProducesSameOutput) {
    auto token_ids = torch::randint(kVocabSize, {4}, torch::kLong);
    auto out1 = emb_->forward(token_ids);
    auto out2 = emb_->forward(token_ids);
    EXPECT_TRUE(out1.equal(out2));
}

// ============================================================================
// Gradient flow test
// ============================================================================

TEST_F(EmbeddingLayerTest, GradientsFlowThroughEmbeddings) {
    auto token_ids = torch::randint(kVocabSize, {2, 4}, torch::kLong);
    auto out = emb_->forward(token_ids);
    auto loss = out.sum();
    loss.backward();

    // Both embedding tables should have accumulated gradients
    EXPECT_TRUE(emb_->token_emb->weight.grad().defined());
    EXPECT_TRUE(emb_->pos_emb->weight.grad().defined());
}

// ============================================================================
// Registered parameters test
// ============================================================================

TEST_F(EmbeddingLayerTest, CorrectNumberOfParameters) {
    // token_emb: vocab_size * emb_dim  +  pos_emb: context_length * emb_dim
    int64_t expected = kVocabSize * kEmbeddingDim + kContextLength * kEmbeddingDim;
    int64_t actual = 0;
    for (const auto& p : emb_->parameters()) {
        actual += p.numel();
    }
    EXPECT_EQ(actual, expected);
}

} // namespace testing
} // namespace llm
