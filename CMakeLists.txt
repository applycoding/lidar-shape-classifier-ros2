cmake_minimum_required(VERSION 3.16)
project(lidar_shape_classifier VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The core, the file CLI, and the scene generator always build.
# The ROS 2 node is added when ament and the Humble message packages are found.
option(BUILD_ROS_NODE "Build the ROS 2 wrapper when ROS 2 is available" ON)

if(BUILD_ROS_NODE)
  find_package(ament_cmake QUIET)
endif()

if(NOT DEFINED BUILD_TESTING)
  set(BUILD_TESTING ON)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PCL 1.10 REQUIRED COMPONENTS
  common
  filters
  io
  kdtree
  sample_consensus
  search
  segmentation
)

add_library(lsc_warnings INTERFACE)
target_compile_options(lsc_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror=switch)

add_library(lidar_shape_core STATIC
  src/synthetic.cpp
  src/preprocess.cpp
  src/ground.cpp
  src/cluster.cpp
  src/classify.cpp
  src/pipeline.cpp
  src/io.cpp
)
target_include_directories(lidar_shape_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(lidar_shape_core SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_options(lidar_shape_core PRIVATE ${PCL_DEFINITIONS} -Werror)
target_compile_definitions(lidar_shape_core PRIVATE EIGEN_HAS_CXX17_OVERALIGN=0)
target_link_libraries(lidar_shape_core
  PUBLIC Eigen3::Eigen
  PRIVATE ${PCL_LIBRARIES} lsc_warnings
)
set_target_properties(lidar_shape_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(shape_classify src/cli_main.cpp)
target_link_libraries(shape_classify PRIVATE lidar_shape_core lsc_warnings)
target_compile_options(shape_classify PRIVATE -Werror)

add_executable(generate_scene src/generate_scene_main.cpp)
target_link_libraries(generate_scene PRIVATE lidar_shape_core lsc_warnings)
target_compile_options(generate_scene PRIVATE -Werror)

if(BUILD_TESTING)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(lidar_shape_core_tests test/test_core.cpp)
  target_link_libraries(lidar_shape_core_tests PRIVATE lidar_shape_core GTest::gtest_main lsc_warnings)
  target_compile_options(lidar_shape_core_tests PRIVATE -Werror)
  add_test(NAME lidar_shape_core_tests COMMAND lidar_shape_core_tests)
endif()

set(LSC_HAS_ROS FALSE)
if(BUILD_ROS_NODE AND ament_cmake_FOUND)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(pcl_conversions REQUIRED)
  set(LSC_HAS_ROS TRUE)
endif()

if(LSC_HAS_ROS)
  add_executable(shape_classifier_node src/ros/shape_classifier_node.cpp)
  target_compile_options(shape_classifier_node PRIVATE -Wall -Wextra -Wpedantic -Werror=switch)
  target_link_libraries(shape_classifier_node lidar_shape_core)
  ament_target_dependencies(shape_classifier_node
    rclcpp
    sensor_msgs
    std_msgs
    geometry_msgs
    visualization_msgs
    pcl_conversions
  )
  install(TARGETS shape_classifier_node shape_classify generate_scene
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
  install(DIRECTORY include/ DESTINATION include)
  ament_package()
else()
  install(TARGETS shape_classify generate_scene RUNTIME DESTINATION bin)
  message(STATUS "lidar_shape_classifier: building core, CLI, and scene generator")
endif()
