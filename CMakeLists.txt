cmake_minimum_required(VERSION 2.8)

PROJECT(VisualIntelligence)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(OpenCV 3.0.0 REQUIRED)

# Modify the following line to add new files to your project 
add_executable(visualIntelligence ./src/main.cpp ./src/freenect-playback-wrapper.h ./src/freenect-playback-wrapper.cpp)
 
target_link_libraries(visualIntelligence ${OpenCV_LIBS})
