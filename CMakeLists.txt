cmake_minimum_required(VERSION 3.16.3)
#添加对CUDA的支持
project(RT_CUDA LANGUAGES CUDA CXX)

find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(USE_CUDA)
else(CUDA_FOUND)
    message(STATUS "No CUDA support")
    remove_definitions(USE_CUDA)
endif(CUDA_FOUND)

set(INCLUDE_PATH "./include")
include_directories(${INCLUDE_PATH})

#读取当前目录下所有源码
aux_source_directory("." M_SRC)
aux_source_directory("./src" SRC)

#读取所有头文件
file(GLOB H_FILE "${INCLUDE_PATH}/*.h*")
set(CMAKE_CUDA_ARCHITECTURES 86)
add_executable(${PROJECT_NAME} ${M_SRC} ${SRC} ${H_FILE})

##set(src  kernel.cu)
#set(CMAKE_CUDA_ARCHITECTURES 86)
##add_executable(${PROJECT_NAME} ${src})