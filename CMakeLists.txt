cmake_minimum_required(VERSION 3.16)
project(mlp_for_mnist)

set(CMAKE_CXX_STANDARD 14)

add_library(mlp_for_mnist mnistrec.cpp mnistrec.h)