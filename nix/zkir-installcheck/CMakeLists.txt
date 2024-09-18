cmake_minimum_required(VERSION 3.20)
project(ZKIRInstallCheck)

find_package(ZKIR REQUIRED CONFIG)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE ZKIR::ZKIRDialect)