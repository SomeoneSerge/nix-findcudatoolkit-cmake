cmake_minimum_required(VERSION 3.25)
project(SomeLib LANGUAGES CXX)

add_library(somelib some.cc)
target_include_directories(
  somelib PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
                 "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

include(GNUInstallDirs)

install(TARGETS somelib EXPORT somelibTargets)
install(
  EXPORT somelibTargets
  FILE SomeLibConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SomeLib
  NAMESPACE SomeLib::)

file(GLOB SOMELIB_HEADERS CONFIGURE_DEPENDS "*.h" "*.hpp")
install(FILES ${SOMELIB_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
