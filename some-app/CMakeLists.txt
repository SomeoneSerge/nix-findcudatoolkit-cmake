cmake_minimum_required(VERSION 3.25)
project(SomeApp LANGUAGES CXX)

find_package(SomeLib REQUIRED)

add_executable(someapp main.cc)
target_link_libraries(someapp somelib)

install(TARGETS someapp)
