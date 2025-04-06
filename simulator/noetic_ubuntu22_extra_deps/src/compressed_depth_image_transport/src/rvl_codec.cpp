// The following code is a C++ wrapper of the code presented by
// Andrew D. Wilson in "Fast Lossless Depth Image Compression" at SIGCHI'17.
// The original code is licensed under the MIT License.

#include "compressed_depth_image_transport/rvl_codec.h"

namespace compressed_depth_image_transport {

RvlCodec::RvlCodec() {}

void RvlCodec::EncodeVLE(int value) {
  do {
    int nibble = value & 0x7;        // lower 3 bits
    if (value >>= 3) nibble |= 0x8;  // more to come
    word_ <<= 4;
    word_ |= nibble;
    if (++nibblesWritten_ == 8)  // output word
    {
      *pBuffer_++ = word_;
      nibblesWritten_ = 0;
      word_ = 0;
    }
  } while (value);
}

int RvlCodec::DecodeVLE() {
  unsigned int nibble;
  int value = 0, bits = 29;
  do {
    if (!nibblesWritten_) {
      word_ = *pBuffer_++;  // load word
      nibblesWritten_ = 8;
    }
    nibble = word_ & 0xf0000000;
    value |= (nibble << 1) >> bits;
    word_ <<= 4;
    nibblesWritten_--;
    bits -= 3;
  } while (nibble & 0x80000000);
  return value;
}

int RvlCodec::CompressRVL(const unsigned short* input, unsigned char* output,
                          int numPixels) {
  buffer_ = pBuffer_ = (int*)output;
  nibblesWritten_ = 0;
  const unsigned short* end = input + numPixels;
  unsigned short previous = 0;
  while (input != end) {
    int zeros = 0, nonzeros = 0;
    for (; (input != end) && !*input; input++, zeros++)
      ;
    EncodeVLE(zeros);  // number of zeros
    for (const unsigned short* p = input; (p != end) && *p++; nonzeros++)
      ;
    EncodeVLE(nonzeros);  // number of nonzeros
    for (int i = 0; i < nonzeros; i++) {
      unsigned short current = *input++;
      int delta = current - previous;
      int positive = (delta << 1) ^ (delta >> 31);
      EncodeVLE(positive);  // nonzero value
      previous = current;
    }
  }
  if (nibblesWritten_)  // last few values
    *pBuffer_++ = word_ << 4 * (8 - nibblesWritten_);
  return int((unsigned char*)pBuffer_ - (unsigned char*)buffer_);  // num bytes
}

void RvlCodec::DecompressRVL(const unsigned char* input, unsigned short* output,
                             int numPixels) {
  buffer_ = pBuffer_ = const_cast<int*>(reinterpret_cast<const int*>(input));
  nibblesWritten_ = 0;
  unsigned short current, previous = 0;
  int numPixelsToDecode = numPixels;
  while (numPixelsToDecode) {
    int zeros = DecodeVLE();  // number of zeros
    numPixelsToDecode -= zeros;
    for (; zeros; zeros--) *output++ = 0;
    int nonzeros = DecodeVLE();  // number of nonzeros
    numPixelsToDecode -= nonzeros;
    for (; nonzeros; nonzeros--) {
      int positive = DecodeVLE();  // nonzero value
      int delta = (positive >> 1) ^ -(positive & 1);
      current = previous + delta;
      *output++ = current;
      previous = current;
    }
  }
}

}  // namespace compressed_depth_image_transport
