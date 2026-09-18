cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(deepfin_bend_aoti_probe LANGUAGES C CXX)

if(NOT DEFINED BEND_GENERATED_C)
  message(FATAL_ERROR "BEND_GENERATED_C must point to Bend-emitted C")
endif()

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

add_executable(
  deepfin_bend_aoti_probe
  "${BEND_GENERATED_C}"
  "${CMAKE_CURRENT_SOURCE_DIR}/aoti_bridge.cpp"
)

set_property(TARGET deepfin_bend_aoti_probe PROPERTY C_STANDARD 11)
set_property(TARGET deepfin_bend_aoti_probe PROPERTY CXX_STANDARD 20)
set_property(TARGET deepfin_bend_aoti_probe PROPERTY CXX_STANDARD_REQUIRED ON)

target_compile_options(
  deepfin_bend_aoti_probe
  PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_link_libraries(
  deepfin_bend_aoti_probe
  PRIVATE
  "${TORCH_LIBRARIES}"
  Threads::Threads
  m
)
