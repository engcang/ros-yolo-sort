cmake_minimum_required(VERSION 3.0.2)
project(sort_ros)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    darknet_ros_msgs
)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()


catkin_package(
  CATKIN_DEPENDS
    roscpp
    darknet_ros_msgs
)

include_directories(
${CMAKE_SOURCE_DIR}/src
${PROJECT_SOURCE_DIR}/src
${OPENCV_INCLUDE_DIR}
${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME} SHARED
src/KalmanTracker.cpp
src/Hungarian.cpp
)
## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
)


### Build ###

## Declare a C++ executable
add_executable(${PROJECT_NAME}_node src/ros_main.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME} ${OpenCV_LIBS} ${catkin_LIBRARIES})
