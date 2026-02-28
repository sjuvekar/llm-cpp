/**
 * @file dataloader_test.cpp
 * @brief Unit and integration tests for dataloader.h utilities
 *
 * @author Sudeep Juvekar (sjuvekar@gmail.com)
 */

#include <gtest/gtest.h>
#include "dataloader.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace llm {
namespace testing {

// ============================================================================
// DataLoaderConfig default value tests
// ============================================================================

TEST(DataLoaderConfigTest, DefaultValues) {
    DataLoaderConfig config;
    EXPECT_EQ(config.batch_size, 4u);
    EXPECT_EQ(config.max_length, 256u);
    EXPECT_EQ(config.stride, 128u);
    EXPECT_TRUE(config.shuffle);
    EXPECT_TRUE(config.drop_last);
    EXPECT_EQ(config.num_workers, 0u);
    EXPECT_EQ(config.language_model, LanguageModel::R50K_BASE);
}

// ============================================================================
// read_text_file tests
// ============================================================================

TEST(ReadTextFileTest, ThrowsOnNonexistentFile) {
    EXPECT_THROW(read_text_file("/nonexistent/path/to/file.txt"), std::runtime_error);
}

TEST(ReadTextFileTest, ReturnsFileContents) {
    auto tmp_path = (std::filesystem::temp_directory_path() / "llm_read_test.txt").string();
    std::string expected = "Hello, world!\nThis is a test.\n";

    { std::ofstream out(tmp_path); out << expected; }

    EXPECT_EQ(read_text_file(tmp_path), expected);

    std::filesystem::remove(tmp_path);
}

// ============================================================================
// create_dataloader integration tests (require tiktoken library installed)
// ============================================================================

// Enough tokens to produce multiple batches with small window/stride settings.
static const std::string kTestText =
    "In the beginning God created the heavens and the earth. "
    "Now the earth was formless and empty, darkness was over the surface of the deep, "
    "and the Spirit of God was hovering over the waters. And God said, let there be light, "
    "and there was light. God saw that the light was good, and he separated the light from "
    "the darkness. God called the light day and the darkness he called night. And there was "
    "evening and there was morning, the first day.";

class CreateDataLoaderTest : public ::testing::Test {
protected:
    DataLoaderConfig small_config() {
        DataLoaderConfig cfg;
        cfg.max_length = 4;
        cfg.stride = 2;
        cfg.batch_size = 2;
        cfg.shuffle = false;
        cfg.drop_last = true;
        cfg.num_workers = 0;
        return cfg;
    }
};

TEST_F(CreateDataLoaderTest, ReturnsNonNullDataLoader) {
    auto dataloader = create_dataloader(kTestText, small_config());
    EXPECT_NE(dataloader, nullptr);
}

TEST_F(CreateDataLoaderTest, BatchHasCorrectShape) {
    auto cfg = small_config();
    auto dataloader = create_dataloader(kTestText, cfg);

    bool got_batch = false;
    for (auto& batch : *dataloader) {
        EXPECT_EQ(batch.data.dim(), 2);
        EXPECT_EQ(batch.data.size(0), static_cast<int64_t>(cfg.batch_size));
        EXPECT_EQ(batch.data.size(1), static_cast<int64_t>(cfg.max_length));
        EXPECT_EQ(batch.target.dim(), 2);
        EXPECT_EQ(batch.target.size(0), static_cast<int64_t>(cfg.batch_size));
        EXPECT_EQ(batch.target.size(1), static_cast<int64_t>(cfg.max_length));
        got_batch = true;
        break;
    }
    EXPECT_TRUE(got_batch);
}

TEST_F(CreateDataLoaderTest, BatchTensorDTypes) {
    auto dataloader = create_dataloader(kTestText, small_config());

    for (auto& batch : *dataloader) {
        EXPECT_EQ(batch.data.dtype(), torch::kInt64);
        EXPECT_EQ(batch.target.dtype(), torch::kInt64);
        break;
    }
}

TEST_F(CreateDataLoaderTest, DropLastDropsPartialBatches) {
    auto cfg = small_config();
    auto dataloader = create_dataloader(kTestText, cfg);

    for (auto& batch : *dataloader) {
        // Every batch must be exactly batch_size with drop_last=true
        EXPECT_EQ(batch.data.size(0), static_cast<int64_t>(cfg.batch_size));
    }
}

TEST_F(CreateDataLoaderTest, TargetIsInputShiftedByOne) {
    DataLoaderConfig cfg;
    cfg.max_length = 8;
    cfg.stride = 1;
    cfg.batch_size = 1;
    cfg.shuffle = false;
    cfg.drop_last = false;
    cfg.num_workers = 0;

    auto dataloader = create_dataloader(kTestText, cfg);

    for (auto& batch : *dataloader) {
        auto input  = batch.data[0];    // [max_length]
        auto target = batch.target[0];  // [max_length]

        // input[1:] should equal target[:-1]
        auto input_tail   = input.slice(0, 1);
        auto target_head  = target.slice(0, 0, static_cast<int64_t>(cfg.max_length) - 1);
        EXPECT_TRUE(input_tail.equal(target_head));
        break;
    }
}

// ============================================================================
// create_dataloader_from_file tests
// ============================================================================

TEST(CreateDataLoaderFromFileTest, ThrowsOnNonexistentFile) {
    EXPECT_THROW(
        create_dataloader_from_file("/nonexistent/path/file.txt"),
        std::runtime_error
    );
}

TEST(CreateDataLoaderFromFileTest, ProducesDataLoaderFromFile) {
    auto tmp_path = (std::filesystem::temp_directory_path() / "llm_dl_test.txt").string();

    { std::ofstream out(tmp_path); out << kTestText; }

    DataLoaderConfig cfg;
    cfg.max_length = 4;
    cfg.stride = 2;
    cfg.batch_size = 2;
    cfg.shuffle = false;
    cfg.drop_last = true;
    cfg.num_workers = 0;

    auto dataloader = create_dataloader_from_file(tmp_path, cfg);
    EXPECT_NE(dataloader, nullptr);

    bool got_batch = false;
    for (auto& batch : *dataloader) {
        EXPECT_EQ(batch.data.size(0), static_cast<int64_t>(cfg.batch_size));
        EXPECT_EQ(batch.data.size(1), static_cast<int64_t>(cfg.max_length));
        got_batch = true;
        break;
    }
    EXPECT_TRUE(got_batch);

    std::filesystem::remove(tmp_path);
}

} // namespace testing
} // namespace llm
