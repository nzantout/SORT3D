cmake_minimum_required(VERSION 3.0.2)
project(compressed_depth_image_transport)

if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
endif()

find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED cv_bridge dynamic_reconfigure image_transport)

# generate the dynamic_reconfigure config file
generate_dynamic_reconfigure_options(cfg/CompressedDepthPublisher.cfg)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS cv_bridge dynamic_reconfigure image_transport
  DEPENDS OpenCV
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES src/compressed_depth_publisher.cpp src/compressed_depth_subscriber.cpp src/manifest.cpp src/codec.cpp src/rvl_codec.cpp)
add_library(${PROJECT_NAME} ${SOURCE_FILES})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

class_loader_hide_library_symbols(${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

# add xml file
install(FILES compressed_depth_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  # Build ${PROJECT_NAME}_test library with symbols exported.
  add_library(${PROJECT_NAME}_test ${SOURCE_FILES})
  add_dependencies(${PROJECT_NAME}_test ${PROJECT_NAME}_gencfg)
  target_link_libraries(${PROJECT_NAME}_test ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

  catkin_add_gtest(rvl_codec_test test/rvl_codec_test.cpp)
  target_link_libraries(rvl_codec_test ${PROJECT_NAME}_test)
endif()
