/**
 * @file dataset_test.cpp
 * @brief Unit tests for GPTDataset using mock tokenizer
 *
 * @author Sudeep Juvekar (sjuvekar@gmail.com)
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "dataset.h"
#include "mocks/mock_tokenizer.h"

using ::testing::_;
using ::testing::Return;

namespace llm {
namespace testing {

/**
 * @brief Test fixture for GPTDataset tests
 */
class GPTDatasetTest : public ::testing::Test {
protected:
    void SetUp() override {
        tokenizer_ = std::make_shared<MockTokenizer>();
    }

    void TearDown() override {
        tokenizer_.reset();
    }

    std::shared_ptr<MockTokenizer> tokenizer_;
};

/**
 * @test BasicDatasetCreation
 * @brief Verify that dataset creates correct number of samples
 *
 * With 10 tokens and window_length=4, stride=1:
 * - We can create 10 - 4 = 6 samples (indices 0-5)
 * - Each sample has input of length 4 and target of length 4
 */
TEST_F(GPTDatasetTest, BasicDatasetCreation) {
    // Mock tokenizer returns 10 tokens
    std::vector<int> mock_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_CALL(*tokenizer_, encode(_, _))
        .WillOnce(Return(mock_tokens));

    GPTDataset dataset("test text", /*window_length=*/4, /*stride=*/1, tokenizer_);

    // With 10 tokens and window_length=4, stride=1:
    // Number of samples = (10 - 4) / 1 = 6
    EXPECT_EQ(dataset.size().value(), 6u);
}

/**
 * @test InputTargetShift
 * @brief Verify that target is shifted by 1 from input
 *
 * For tokens [1, 2, 3, 4, 5, 6]:
 * - Sample 0: input=[1, 2, 3, 4], target=[2, 3, 4, 5]
 * - Sample 1: input=[2, 3, 4, 5], target=[3, 4, 5, 6]
 */
TEST_F(GPTDatasetTest, InputTargetShift) {
    std::vector<int> mock_tokens = {1, 2, 3, 4, 5, 6};
    EXPECT_CALL(*tokenizer_, encode(_, _))
        .WillOnce(Return(mock_tokens));

    GPTDataset dataset("test text", /*window_length=*/4, /*stride=*/1, tokenizer_);

    // Check first sample
    auto sample0 = dataset.get(0);
    auto input0 = sample0.data.to(torch::kInt64);
    auto target0 = sample0.target.to(torch::kInt64);

    std::vector<int64_t> expected_input0 = {1, 2, 3, 4};
    std::vector<int64_t> expected_target0 = {2, 3, 4, 5};

    auto input0_vec = std::vector<int64_t>(input0.data_ptr<int64_t>(), input0.data_ptr<int64_t>() + input0.numel());
    auto target0_vec = std::vector<int64_t>(target0.data_ptr<int64_t>(), target0.data_ptr<int64_t>() + target0.numel());

    EXPECT_EQ(input0_vec, expected_input0);
    EXPECT_EQ(target0_vec, expected_target0);
}

/**
 * @test StrideOverlap
 * @brief Verify correct overlap when stride < window_length
 *
 * For tokens [1, 2, 3, 4, 5, 6, 7, 8] with window_length=4, stride=2:
 * - Sample 0: input=[1, 2, 3, 4], target=[2, 3, 4, 5]
 * - Sample 1: input=[3, 4, 5, 6], target=[4, 5, 6, 7]
 * - Sample 2: input=[5, 6, 7, 8], target=[6, 7, 8, 9] (if there were 9 tokens)
 *
 * With 8 tokens: samples = (8 - 4) / 2 = 2 samples
 */
TEST_F(GPTDatasetTest, StrideOverlap) {
    std::vector<int> mock_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_CALL(*tokenizer_, encode(_, _))
        .WillOnce(Return(mock_tokens));

    GPTDataset dataset("test text", /*window_length=*/4, /*stride=*/2, tokenizer_);

    // (8 - 4) / 2 = 2 samples
    EXPECT_EQ(dataset.size().value(), 2u);

    // Check that sample 1 starts at index 2 (stride=2 from start)
    auto sample1 = dataset.get(1);
    auto input1 = sample1.data.to(torch::kInt64);

    std::vector<int64_t> expected_input1 = {3, 4, 5, 6};
    auto input1_vec = std::vector<int64_t>(input1.data_ptr<int64_t>(), input1.data_ptr<int64_t>() + input1.numel());

    EXPECT_EQ(input1_vec, expected_input1);
}

/**
 * @test StrideEqualsWindow
 * @brief Verify no overlap when stride == window_length
 *
 * For tokens [1, 2, 3, 4, 5, 6, 7, 8] with window_length=4, stride=4:
 * - Sample 0: input=[1, 2, 3, 4], target=[2, 3, 4, 5]
 * - Sample 1: input=[5, 6, 7, 8], target=[6, 7, 8, 9] (if there were 9 tokens)
 *
 * With 8 tokens: samples = (8 - 4) / 4 = 1 sample
 */
TEST_F(GPTDatasetTest, StrideEqualsWindow) {
    std::vector<int> mock_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_CALL(*tokenizer_, encode(_, _))
        .WillOnce(Return(mock_tokens));

    GPTDataset dataset("test text", /*window_length=*/4, /*stride=*/4, tokenizer_);

    // (8 - 4) / 4 = 1 sample
    EXPECT_EQ(dataset.size().value(), 1u);
}

/**
 * @test ShortTextHandling
 * @brief Verify handling of text shorter than window_length
 *
 * With 3 tokens and window_length=4:
 * - (3 - 4) = -1, so no samples should be created
 */
TEST_F(GPTDatasetTest, ShortTextHandling) {
    std::vector<int> mock_tokens = {1, 2, 3};
    EXPECT_CALL(*tokenizer_, encode(_, _))
        .WillOnce(Return(mock_tokens));

    GPTDataset dataset("short", /*window_length=*/4, /*stride=*/1, tokenizer_);

    // Not enough tokens to create even one sample
    EXPECT_EQ(dataset.size().value(), 0u);
}

/**
 * @test ExactWindowSize
 * @brief Verify handling when token count equals window_length + 1
 *
 * With 5 tokens and window_length=4:
 * - Exactly one sample can be created
 * - input=[1, 2, 3, 4], target=[2, 3, 4, 5]
 */
TEST_F(GPTDatasetTest, ExactWindowSize) {
    std::vector<int> mock_tokens = {1, 2, 3, 4, 5};
    EXPECT_CALL(*tokenizer_, encode(_, _))
        .WillOnce(Return(mock_tokens));

    GPTDataset dataset("exact", /*window_length=*/4, /*stride=*/1, tokenizer_);

    // Exactly one sample
    EXPECT_EQ(dataset.size().value(), 1u);
}

/**
 * @test TensorDataTypes
 * @brief Verify that tensors have correct data types (Int64)
 */
TEST_F(GPTDatasetTest, TensorDataTypes) {
    std::vector<int> mock_tokens = {1, 2, 3, 4, 5, 6};
    EXPECT_CALL(*tokenizer_, encode(_, _))
        .WillOnce(Return(mock_tokens));

    GPTDataset dataset("test", /*window_length=*/4, /*stride=*/1, tokenizer_);

    auto sample = dataset.get(0);

    // Check that tensors are Int64 (Long)
    EXPECT_EQ(sample.data.dtype(), torch::kInt64);
    EXPECT_EQ(sample.target.dtype(), torch::kInt64);
}

/**
 * @test TensorShapes
 * @brief Verify that tensors have correct shapes
 */
TEST_F(GPTDatasetTest, TensorShapes) {
    std::vector<int> mock_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_CALL(*tokenizer_, encode(_, _))
        .WillOnce(Return(mock_tokens));

    int64_t window_length = 4;
    GPTDataset dataset("test", window_length, /*stride=*/1, tokenizer_);

    auto sample = dataset.get(0);

    // Check tensor shapes (1D with size = window_length)
    EXPECT_EQ(sample.data.dim(), 1);
    EXPECT_EQ(sample.data.size(0), window_length);
    EXPECT_EQ(sample.target.dim(), 1);
    EXPECT_EQ(sample.target.size(0), window_length);
}

} // namespace testing
} // namespace llm
