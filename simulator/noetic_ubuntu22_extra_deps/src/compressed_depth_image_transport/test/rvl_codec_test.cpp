#include "compressed_depth_image_transport/rvl_codec.h"
#include <gtest/gtest.h>

TEST(RvlCodecTest, reciprocalTestEmpty) {
  const int size = 1000000;
  std::vector<unsigned short> original(size);
  std::vector<unsigned char> compressed(3 * size + 4);
  std::vector<unsigned short> decompressed(size);
  compressed_depth_image_transport::RvlCodec rvl;

  // Constant depth.
  const int validDepth = 42;
  std::fill(original.begin(), original.end(), validDepth);
  rvl.CompressRVL(&original[0], &compressed[0], size);
  rvl.DecompressRVL(&compressed[0], &decompressed[0], size);
  EXPECT_TRUE(std::equal(original.begin(), original.end(), decompressed.begin()));

  // Totally invalid depth.
  const int invalidDepth = 0;
  std::fill(original.begin(), original.end(), invalidDepth);
  rvl.CompressRVL(&original[0], &compressed[0], size);
  rvl.DecompressRVL(&compressed[0], &decompressed[0], size);
  EXPECT_TRUE(std::equal(original.begin(), original.end(), decompressed.begin()));

  // Empty depth.
  EXPECT_EQ(rvl.CompressRVL(NULL, NULL, 0), 0);
  rvl.DecompressRVL(NULL, NULL, 0);  // should not die.
}

TEST(RvlCodecTest, reciprocalTestRandom) {
  const int size = 1000000;
  std::vector<unsigned short> original(size);
  std::vector<unsigned char> compressed(3 * size + 4);
  std::vector<unsigned short> decompressed(size);
  compressed_depth_image_transport::RvlCodec rvl;

  // Populate depths with random size of runs with random values.
  for (int i = 0; i < size;) {
    int length = std::min<int>(rand() % 10, size - i);
    int value = rand() % 10;
    std::fill(&original[i], &original[i] + length, value);
    i += length;
  }

  const int compressedSize =
      rvl.CompressRVL(&original[0], &compressed[0], size);
  EXPECT_GT(compressedSize, 0);
  EXPECT_LT(compressedSize, compressed.size());
  rvl.DecompressRVL(&compressed[0], &decompressed[0], size);
  EXPECT_TRUE(std::equal(original.begin(), original.end(), decompressed.begin()));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
