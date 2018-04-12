cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project("FaceDeploy")

###============= C++11 support ==================================
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()
#================================================================

#=============== Find Packages ==================================
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
  message (STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
  message (STATUS "OpenCV_LIBRARIES=${OpenCV_LIBRARIES}")
endif()

list(APPEND CMAKE_PREFIX_PATH /home/yf/.software/caffe-1.0)
find_package(Caffe REQUIRED)
if (Caffe_FOUND)
  message (STATUS "Caffe_LIBRARIES=${Caffe_LIBRARIES}")
endif()

file(GLOB PROJECT_CORE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/core/*.h)
file(GLOB PROJECT_CORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp)
file(GLOB PROJECT_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)

foreach(src ${PROJECT_TEST_SRC})
  # get filename
  get_filename_component(name ${src} NAME_WE)
  # get folder name
  get_filename_component(path ${src} PATH)
  get_filename_component(folder ${path} NAME_WE)

  # build
  add_executable(${name} ${src} 
    ${PROJECT_CORE_SRC} ${PROJECT_CORE_INCLUDE})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${name}
    ${OpenCV_LIBRARIES} ${Caffe_LIBRARIES})
  
  # install
  install(TARGETS ${name} DESTINATION bin)
endforeach()



