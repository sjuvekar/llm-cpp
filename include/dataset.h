/**
 * @file dataset.h
 * @brief C++ implementation of GPTDataset
 *
 * Dependencies:
 * - libtorch: PyTorch C++ API for tensors
 * - ITokenizer: Abstract tokenizer interface
 *
 * @author Sudeep Juvekar (sjuvekar@gmail.com)
 */
#ifndef LLMS_FROM_SCRATCH_DATASET_H
#define LLMS_FROM_SCRATCH_DATASET_H

#include <torch/torch.h>
#include <tiktoken/encoding.h>
#include <string>
#include <unordered_set>
#include <vector>
#include <memory>
#include "tokenizer.h"

namespace llm {

/**
 * @class GPTDataset
 * @brief Dataset class that creates overlapping input/target sequences from text
 *
 * This dataset tokenizes the entire text using a BPE tokenizer, then creates
 * overlapping chunks using a sliding window approach. Each sample consists of:
 * - input_ids: A sequence of token IDs
 * - target_ids: The same sequence shifted by one position (next-token prediction)
 *
 * Example with window_length=4, stride=1:
 *   Text: "The quick brown fox"
 *   Token IDs: [1, 2, 3, 4, 5]
 *
 *   Sample 0: input=[1, 2, 3, 4], target=[2, 3, 4, 5]
 *   Sample 1: input=[2, 3, 4, 5], target=[3, 4, 5, 6]
 *   ...
 */
class GPTDataset : public torch::data::Dataset<GPTDataset> {
public:
    /**
     * @brief Construct a GPTDataset with a custom tokenizer
     *
     * @param txt The raw text to tokenize and create sequences from
     * @param window_length The length of each sequence (context length)
     * @param stride The step size for the sliding window (overlap = window_length - stride)
     * @param tokenizer Tokenizer to use for encoding text
     * @param allowed_special allowed special characters during encoding
     *
     * @throws std::runtime_error if text is too short for window_length
     */
    GPTDataset(
        const std::string& txt,
        int64_t window_length,
        int64_t stride,
        std::shared_ptr<ITokenizer> tokenizer,
        const std::unordered_set<std::string>& allowed_special = {}
    );

    /**
     * @brief Get a single sample from the dataset
     *
     * @param index The sample index
     * @return A pair of (input_ids, target_ids) tensors
     */
    torch::data::Example<> get(size_t index) override {
        return {input_ids_[index], target_ids_[index]};
    }

    /**
     * @brief Get the number of samples in the dataset
     */
    torch::optional<size_t> size() const override {
        return input_ids_.size();
    }

private:
    void build_dataset();

    int64_t window_length_;                           ///< Context length
    int64_t stride_;                                  ///< Sliding window step size
    std::shared_ptr<ITokenizer> tokenizer_;           ///< Tokenizer interface
    std::unordered_set<std::string> allowed_special_; ///< Special tokens
    std::vector<int> token_ids_;                      ///< All token IDs from the text
    std::vector<torch::Tensor> input_ids_;            ///< Input sequences
    std::vector<torch::Tensor> target_ids_;           ///< Target sequences (shifted by 1)
};

} // namespace llm

#endif //LLMS_FROM_SCRATCH_DATASET_H