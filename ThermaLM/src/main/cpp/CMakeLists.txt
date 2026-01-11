cmake_minimum_required(VERSION 3.22.1)

project("thermalm")

# Set llama.cpp directory (assuming it will be a subdirectory of cpp)
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# Add llama.cpp as a subdirectory if it exists
if (EXISTS ${LLAMA_DIR}/CMakeLists.txt)
    add_subdirectory(${LLAMA_DIR})
endif()

add_library(thermalm SHARED
        native-lib.cpp)

target_include_directories(thermalm PRIVATE
        ${LLAMA_DIR}/include
        ${LLAMA_DIR})

target_link_libraries(thermalm
        # llama is the library name from llama.cpp's CMakeLists.txt
        llama
        log)
