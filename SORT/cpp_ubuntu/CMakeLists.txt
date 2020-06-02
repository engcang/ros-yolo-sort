cmake_minimum_required(VERSION 3.0.2)
project(sort_cpp)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(catkin REQUIRED)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES sort_cpp
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
${CMAKE_SOURCE_DIR}/src
${PROJECT_SOURCE_DIR}/src
${OPENCV_INCLUDE_DIR}
# include
# ${catkin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
## Declare a C++ executable
add_executable(${PROJECT_NAME}_node src/main.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME} ${OpenCV_LIBS})
