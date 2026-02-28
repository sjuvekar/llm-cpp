/**
 * @file tokenizer.h
 * @brief Tokenizer interface for text-to-token encoding
 *
 * This abstraction allows mocking the tokenizer for unit tests
 * while using the real tiktoken implementation in production.
 *
 * @author Sudeep Juvekar (sjuvekar@gmail.com)
 */

#ifndef LLM_TOKENIZER_H
#define LLM_TOKENIZER_H

#include <string>
#include <unordered_set>
#include <vector>
#include <memory>
#include <tiktoken/encoding.h>

namespace llm {

/**
 * @class ITokenizer
 * @brief Abstract interface for text tokenizers
 *
 * This interface abstracts the tokenization process, allowing:
 * - Production use with TiktokenTokenizer (wraps GptEncoding)
 * - Testing with MockTokenizer (GoogleMock)
 */
class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    /**
     * @brief Encode text into token IDs
     *
     * @param text The text to encode
     * @param allowed_special Set of special tokens to allow in encoding
     * @return Vector of token IDs
     */
    virtual std::vector<int> encode(
        const std::string& text,
        const std::unordered_set<std::string>& allowed_special = {}
    ) = 0;
};

/**
 * @brief Factory function to create a TiktokenTokenizer
 *
 * @param model The language model to use for encoding
 * @return Shared pointer to ITokenizer
 */
std::shared_ptr<ITokenizer> create_tiktoken_tokenizer(LanguageModel model);

} // namespace llm

#endif // LLM_TOKENIZER_H