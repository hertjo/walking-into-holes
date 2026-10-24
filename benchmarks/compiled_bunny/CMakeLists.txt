cmake_minimum_required(VERSION 3.18)
project(wih_twos_embree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(embree 4 CONFIG REQUIRED)
find_package(OpenMP)

add_executable(wih_twos src/wih_twos.cpp)
target_link_libraries(wih_twos PRIVATE embree)
target_compile_options(wih_twos PRIVATE -O3 -DNDEBUG -Wall -Wextra -Wpedantic)

if(OpenMP_CXX_FOUND)
  target_link_libraries(wih_twos PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(wih_twos PRIVATE WIH_HAS_OPENMP=1)
endif()

install(TARGETS wih_twos RUNTIME DESTINATION bin)
