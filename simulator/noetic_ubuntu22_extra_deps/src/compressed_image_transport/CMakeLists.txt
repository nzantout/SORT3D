cmake_minimum_required(VERSION 3.0.2)
project(compressed_image_transport)

find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS cv_bridge dynamic_reconfigure image_transport)

find_package(PkgConfig REQUIRED)
pkg_check_modules(TurboJPEG REQUIRED libturbojpeg)

# generate the dynamic_reconfigure config file
generate_dynamic_reconfigure_options(cfg/CompressedPublisher.cfg cfg/CompressedSubscriber.cfg)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS cv_bridge dynamic_reconfigure image_transport
  DEPENDS OpenCV
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${TurboJPEG_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/compressed_publisher.cpp src/compressed_subscriber.cpp src/manifest.cpp)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${TurboJPEG_LIBRARIES})

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
install(FILES compressed_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest_gtest(basic_test test/basic.test test/basic.cpp)
  target_link_libraries(basic_test ${catkin_LIBRARIES})
endif()
