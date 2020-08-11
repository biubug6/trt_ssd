cmake_minimum_required(VERSION 3.14)
project(Infer)

set(CMAKE_CXX_STANDARD 11)

# set cuda
find_package(CUDA 10.1 EXACT REQUIRED cudart cublas curand)

find_package(OpenCV REQUIRED opencv_core opencv_highgui opencv_imgproc)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${tensor_root}/include
                    ${CUDA_INCLUDE_DIRS}
                    )

link_directories(${tensor_root}/lib)

aux_source_directory(${CMAKE_SOURCE_DIR}/src/engine SRC)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/detect SRC)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/executa SRC)

add_executable(detect ${SRC})
target_link_libraries(detect
        ${OpenCV_LIBS}
        nvinfer nvinfer_plugin
        ${CUDA_LIBRARIES})
