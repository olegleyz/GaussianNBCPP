project(playground)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11 -O3)

set(sources
    src/main.cpp
    src/classifier.cpp)

add_executable(playground ${sources})