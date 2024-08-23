cmake_minimum_required(VERSION 3.10.0)

project(YOLOV8)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
set(TensorRT_ROOT /home/wanggq/CUDA/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0/TensorRT-8.6.1.6)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TensorRT_ROOT}/include
    )

link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${TensorRT_ROOT}/lib
)

file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(YOLOV8 yolov8_TensorRT.cpp ${SRC_FILES})
target_link_libraries(YOLOV8 ${OpenCV_LIBS} nvinfer nvinfer_plugin nvparsers nvonnxparser cudart cudnn)
