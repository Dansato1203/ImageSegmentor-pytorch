cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
            "Default build type: RelWithDebInfo" FORCE)
endif()
project(field-detect_test)

find_package(Torch REQUIRED
    HINTS "${CMAKE_PREFIX_PATH}/libtorch"
)

set(LINK_WHAT_YOU_USE TRUE)
find_package(OpenCV REQUIRED)
add_executable(fielddetection field-detect_test.cpp)

target_link_libraries(fielddetection
    opencv_core
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
     "${TORCH_LIBRARIES}"
)

set_property(TARGET fielddetection PROPERTY CXX_STANDARD 14)
