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

#include <image_transport/image_transport.h>

#include <gtest/gtest.h>

#include <cv_bridge/cv_bridge.h>

static unsigned int receivedImages = 0;
std::vector<std::string> receivedEncodings;

void handleImage(const sensor_msgs::ImageConstPtr& img)
{
  cv_bridge::CvImagePtr cvImg = cv_bridge::toCvCopy(img, "bgr8");
  receivedEncodings.push_back(img->encoding);

  ASSERT_EQ(cvImg->image.rows, 400);
  ASSERT_EQ(cvImg->image.cols, 400);

  // Red corner
  cv::Scalar upperLeft = cv::mean(cvImg->image(cv::Rect(0, 0, 200, 200)));
  EXPECT_LT(upperLeft[0], 10);
  EXPECT_LT(upperLeft[1], 10);
  EXPECT_GT(upperLeft[2], 240);

  // Green corner
  cv::Scalar upperRight = cv::mean(cvImg->image(cv::Rect(200, 0, 200, 200)));
  EXPECT_LT(upperRight[0], 10);
  EXPECT_GT(upperRight[1], 240);
  EXPECT_LT(upperRight[2], 10);

  // Blue corner
  cv::Scalar lowerRight = cv::mean(cvImg->image(cv::Rect(200, 200, 200, 200)));
  EXPECT_GT(lowerRight[0], 240);
  EXPECT_LT(lowerRight[1], 10);
  EXPECT_LT(lowerRight[2], 10);

  // White corner
  cv::Scalar lowerLeft = cv::mean(cvImg->image(cv::Rect(0, 200, 200, 200)));
  EXPECT_GT(lowerLeft[0], 240);
  EXPECT_GT(lowerLeft[1], 240);
  EXPECT_GT(lowerLeft[2], 240);

  receivedImages++;
}

TEST(Basic, loop)
{
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  image_transport::Publisher pub = it.advertise("img", 20);
  image_transport::Subscriber sub = it.subscribe("img", 20, &handleImage, image_transport::TransportHints("compressed"));

  // Create a nice test image. Corners will be red, green, blue, white in clock-wise order.
  cv::Mat_<cv::Vec3b> testImage(400, 400); // This is BGR!
  for(int y = 0; y < 200; ++y)
  {
    for(int x = 0; x < 200; ++x)
      testImage(y,x) = cv::Vec3b(0, 0, 255);

    for(int x = 200; x < 400; ++x)
      testImage(y,x) = cv::Vec3b(0, 255, 0);
  }
  for(int y = 200; y < 400; ++y)
  {
    for(int x = 0; x < 200; ++x)
      testImage(y,x) = cv::Vec3b(255, 255, 255);

    for(int x = 200; x < 400; ++x)
      testImage(y,x) = cv::Vec3b(255, 0, 0);
  }

  ros::spinOnce();

  unsigned int expectedImages = 0;
  std::vector<std::string> expectedEncodings;

  // Try different formats
  {
    cv_bridge::CvImage cvImg;
    testImage.copyTo(cvImg.image);
    cvImg.encoding = sensor_msgs::image_encodings::BGR8;

    sensor_msgs::ImagePtr msg = cvImg.toImageMsg();
    EXPECT_EQ(msg->encoding, sensor_msgs::image_encodings::BGR8);

    pub.publish(msg);
    expectedImages++;
    expectedEncodings.push_back(sensor_msgs::image_encodings::BGR8);
  }
  {
    cv_bridge::CvImage cvImg;
    cv::cvtColor(testImage, cvImg.image, CV_BGR2RGB);
    cvImg.encoding = sensor_msgs::image_encodings::RGB8;

    sensor_msgs::ImagePtr msg = cvImg.toImageMsg();
    EXPECT_EQ(msg->encoding, sensor_msgs::image_encodings::RGB8);

    pub.publish(msg);
    expectedImages++;
    expectedEncodings.push_back(sensor_msgs::image_encodings::RGB8);
  }
  {
    cv_bridge::CvImage cvImg;
    cv::cvtColor(testImage, cvImg.image, CV_BGR2RGBA);
    cvImg.encoding = sensor_msgs::image_encodings::RGBA8;

    sensor_msgs::ImagePtr msg = cvImg.toImageMsg();
    EXPECT_EQ(msg->encoding, sensor_msgs::image_encodings::RGBA8);

    pub.publish(msg);
    expectedImages++;
    expectedEncodings.push_back(sensor_msgs::image_encodings::RGBA8);
  }
  {
    cv_bridge::CvImage cvImg;
    cv::cvtColor(testImage, cvImg.image, CV_BGR2BGRA);
    cvImg.encoding = sensor_msgs::image_encodings::BGRA8;

    sensor_msgs::ImagePtr msg = cvImg.toImageMsg();
    EXPECT_EQ(msg->encoding, sensor_msgs::image_encodings::BGRA8);

    pub.publish(msg);
    expectedImages++;
    expectedEncodings.push_back(sensor_msgs::image_encodings::BGRA8);
  }

  ros::WallTime start = ros::WallTime::now();
  while(receivedImages < expectedImages && (ros::WallTime::now() - start) < ros::WallDuration(3.0))
  {
    ros::spinOnce();
    ros::WallDuration(0.1).sleep();
  }
  EXPECT_EQ(receivedImages, expectedImages);

  ASSERT_EQ(receivedEncodings.size(), expectedEncodings.size());
  for(std::size_t i = 0; i < receivedEncodings.size(); ++i)
    EXPECT_EQ(receivedEncodings[i], expectedEncodings[i]);
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "basic");
  ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}
