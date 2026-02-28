/**
 * @file tokenizer.cpp
 * @brief Tiktoken tokenizer implementation
 *
 * Wraps the GptEncoding from cpp-tiktoken library.
 *
 * @author Sudeep Juvekar (sjuvekar@gmail.com)
 */

#include "tokenizer.h"
#include <tiktoken/encoding.h>
#include <stdexcept>

namespace llm {

/**
 * @class TiktokenTokenizer
 * @brief Production tokenizer wrapping GptEncoding from cpp-tiktoken
 */
class TiktokenTokenizer : public ITokenizer {
public:
    /**
     * @brief Construct a TiktokenTokenizer with the specified language model
     *
     * @param model The language model to use for encoding (e.g., R50K_BASE)
     */
    explicit TiktokenTokenizer(LanguageModel model)
        : encoding_(GptEncoding::get_encoding(model))
    {
        if (!encoding_) {
            throw std::runtime_error("Failed to initialize GptEncoding for the specified model");
        }
    }

    /**
     * @brief Encode text into token IDs using tiktoken
     *
     * @param text The text to encode
     * @param allowed_special Set of special tokens to allow in encoding
     * @return Vector of token IDs
     */
    std::vector<int> encode(
        const std::string& text,
        const std::unordered_set<std::string>& allowed_special = {}
    ) override {
        return encoding_->encode(text, allowed_special);
    }

private:
    std::shared_ptr<GptEncoding> encoding_;
};

/**
 * @brief Factory function to create a TiktokenTokenizer
 *
 * @param model The language model to use for encoding
 * @return Shared pointer to ITokenizer
 */
std::shared_ptr<ITokenizer> create_tiktoken_tokenizer(LanguageModel model) {
    return std::make_shared<TiktokenTokenizer>(model);
}

} // namespace llm