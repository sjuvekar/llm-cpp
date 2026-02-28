/**
 * @file mock_tokenizer.h
 * @brief GoogleMock implementation of ITokenizer for unit testing
 *
 * @author Sudeep Juvekar (sjuvekar@gmail.com)
 */

#ifndef LLM_MOCK_TOKENIZER_H
#define LLM_MOCK_TOKENIZER_H

#include <gmock/gmock.h>
#include "tokenizer.h"

namespace llm {
namespace testing {

/**
 * @class MockTokenizer
 * @brief Mock implementation of ITokenizer for unit tests
 *
 * Use this class to control tokenization behavior in tests:
 * @code
 * MockTokenizer mock;
 * EXPECT_CALL(mock, encode("hello", _))
 *     .WillOnce(Return(std::vector<int>{1, 2, 3}));
 * @endcode
 */
class MockTokenizer : public ITokenizer {
public:
    MOCK_METHOD(
        std::vector<int>,
        encode,
        (const std::string& text, const std::unordered_set<std::string>& allowed_special),
        (override)
    );
};

} // namespace testing
} // namespace llm

#endif // LLM_MOCK_TOKENIZER_H