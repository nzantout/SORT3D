/*********************************************************************
* Software License Agreement (BSD License)
* 
*  Copyright (c) 2012, Willow Garage, Inc.
*  All rights reserved.
* 
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
* 
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
* 
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#include <boost/endian/arithmetic.hpp>
#include "compressed_image_transport/compressed_subscriber.h"
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <turbojpeg.h>

#include "compressed_image_transport/compression_common.h"

#include <limits>
#include <vector>

using namespace cv;

namespace enc = sensor_msgs::image_encodings;

namespace compressed_image_transport
{

CompressedSubscriber::CompressedSubscriber()
 : tj_(0)
{}

CompressedSubscriber::~CompressedSubscriber()
{
  if (tj_)
    tjDestroy(tj_);
}

void CompressedSubscriber::subscribeImpl(ros::NodeHandle& nh, const std::string& base_topic, uint32_t queue_size,
                             const Callback& callback, const ros::VoidPtr& tracked_object,
                             const image_transport::TransportHints& transport_hints)
{
    typedef image_transport::SimpleSubscriberPlugin<sensor_msgs::CompressedImage> Base;
    Base::subscribeImpl(nh, base_topic, queue_size, callback, tracked_object, transport_hints);

    // Set up reconfigure server for this topic
    reconfigure_server_ = boost::make_shared<ReconfigureServer>(this->nh());
    ReconfigureServer::CallbackType f = boost::bind(&CompressedSubscriber::configCb, this, boost::placeholders::_1, boost::placeholders::_2);
    reconfigure_server_->setCallback(f);
}


void CompressedSubscriber::configCb(Config& config, uint32_t level)
{
  config_ = config;
  if (config_.mode == compressed_image_transport::CompressedSubscriber_gray) {
      imdecode_flag_ = cv::IMREAD_GRAYSCALE;
  } else if (config_.mode == compressed_image_transport::CompressedSubscriber_color) {
      imdecode_flag_ = cv::IMREAD_COLOR;
  } else /*if (config_.mode == compressed_image_transport::CompressedSubscriber_unchanged)*/ {
      imdecode_flag_ = cv::IMREAD_UNCHANGED;
  } 
}

void CompressedSubscriber::shutdown()
{
  reconfigure_server_.reset();
  image_transport::SimpleSubscriberPlugin<sensor_msgs::CompressedImage>::shutdown();
}

sensor_msgs::ImagePtr CompressedSubscriber::decompressJPEG(const std::vector<uint8_t>& data, const std::string& source_encoding, const std_msgs::Header& header)
{
  if (!tj_)
    tj_ = tjInitDecompress();

  int width, height, jpegSub, jpegColor;

  // Old TurboJPEG require a const_cast here. This was fixed in TurboJPEG 1.5.
  uint8_t* src = const_cast<uint8_t*>(data.data());

  if (tjDecompressHeader3(tj_, src, data.size(), &width, &height, &jpegSub, &jpegColor) != 0)
    return sensor_msgs::ImagePtr(); // If we cannot decode the JPEG header, silently fall back to OpenCV

  sensor_msgs::ImagePtr ret(new sensor_msgs::Image);
  ret->header = header;
  ret->width = width;
  ret->height = height;
  ret->encoding = source_encoding;
  // consistent with cv_bridge
  ret->is_bigendian = (boost::endian::order::native == boost::endian::order::big);  // NOLINT

  int pixelFormat;

  if (source_encoding == enc::MONO8)
  {
    ret->data.resize(height*width);
    ret->step = ret->width;
    pixelFormat = TJPF_GRAY;
  }
  else if (source_encoding == enc::RGB8)
  {
    ret->data.resize(height*width*3);
    ret->step = width*3;
    pixelFormat = TJPF_RGB;
  }
  else if (source_encoding == enc::BGR8)
  {
    ret->data.resize(height*width*3);
    ret->step = width*3;
    pixelFormat = TJPF_BGR;
  }
  else if (source_encoding == enc::RGBA8)
  {
    ret->data.resize(height*width*4);
    ret->step = width*4;
    pixelFormat = TJPF_RGBA;
  }
  else if (source_encoding == enc::BGRA8)
  {
    ret->data.resize(height*width*4);
    ret->step = width*4;
    pixelFormat = TJPF_BGRA;
  }
  else if (source_encoding.empty())
  {
    // Autodetect based on image
    if(jpegColor == TJCS_GRAY)
    {
      ret->data.resize(height*width);
      ret->step = width;
      ret->encoding = enc::MONO8;
      pixelFormat = TJPF_GRAY;
    }
    else
    {
      ret->data.resize(height*width*3);
      ret->step = width*3;
      ret->encoding = enc::RGB8;
      pixelFormat = TJPF_RGB;
    }
  }
  else
  {
    ROS_WARN_THROTTLE(10.0, "Encountered a source encoding that is not supported by TurboJPEG: '%s'", source_encoding.c_str());
    return sensor_msgs::ImagePtr();
  }

  if (tjDecompress2(tj_, src, data.size(), ret->data.data(), width, 0, height, pixelFormat, 0) != 0)
  {
    ROS_WARN_THROTTLE(10.0, "Could not decompress data using TurboJPEG, falling back to OpenCV");
    return sensor_msgs::ImagePtr();
  }

  return ret;
}

void CompressedSubscriber::internalCallback(const sensor_msgs::CompressedImageConstPtr& message,
                                            const Callback& user_cb)

{
  // Parse format field
  std::string image_encoding;
  std::string compressed_encoding;
  {
    const size_t split_pos = message->format.find(';');
    if (split_pos != std::string::npos)
    {
      image_encoding = message->format.substr(0, split_pos);
      compressed_encoding = message->format.substr(split_pos);
    }
  }

  // Try TurboJPEG first (if the first bytes look like JPEG)
  if(message->data.size() > 4 && message->data[0] == 0xFF && message->data[1] == 0xD8)
  {
    sensor_msgs::ImagePtr decoded = decompressJPEG(message->data, image_encoding, message->header);
    if(decoded)
    {
      user_cb(decoded);
      return;
    }
  }

  // Otherwise, try our luck with OpenCV.
  cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

  // Copy message header
  cv_ptr->header = message->header;

  // Decode color/mono image
  try
  {
    cv_ptr->image = cv::imdecode(cv::Mat(message->data), imdecode_flag_);

    // Assign image encoding string
    if (image_encoding.empty())
    {
      // Older version of compressed_image_transport does not signal image format
      switch (cv_ptr->image.channels())
      {
        case 1:
          cv_ptr->encoding = enc::MONO8;
          break;
        case 3:
          cv_ptr->encoding = enc::BGR8;
          break;
        default:
          ROS_ERROR("Unsupported number of channels: %i", cv_ptr->image.channels());
          break;
      }
    } else
    {
      cv_ptr->encoding = image_encoding;

      if ( enc::isColor(image_encoding))
      {
        bool compressed_bgr_image = (compressed_encoding.find("compressed bgr") != std::string::npos);

        // Revert color transformation
        if (compressed_bgr_image)
        {
          // if necessary convert colors from bgr to rgb
          if ((image_encoding == enc::RGB8) || (image_encoding == enc::RGB16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_BGR2RGB);

          if ((image_encoding == enc::RGBA8) || (image_encoding == enc::RGBA16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_BGR2RGBA);

          if ((image_encoding == enc::BGRA8) || (image_encoding == enc::BGRA16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_BGR2BGRA);
        } else
        {
          // if necessary convert colors from rgb to bgr
          if ((image_encoding == enc::BGR8) || (image_encoding == enc::BGR16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_RGB2BGR);

          if ((image_encoding == enc::BGRA8) || (image_encoding == enc::BGRA16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_RGB2BGRA);

          if ((image_encoding == enc::RGBA8) || (image_encoding == enc::RGBA16))
            cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_RGB2RGBA);
        }
      }
    }
  }
  catch (cv::Exception& e)
  {
    ROS_ERROR("%s", e.what());
  }

  size_t rows = cv_ptr->image.rows;
  size_t cols = cv_ptr->image.cols;

  if ((rows > 0) && (cols > 0))
    // Publish message to user callback
    user_cb(cv_ptr->toImageMsg());
}

} //namespace compressed_image_transport
