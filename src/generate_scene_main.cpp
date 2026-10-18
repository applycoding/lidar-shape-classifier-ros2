#include "lidar_shape_classifier/pipeline.hpp"
#include "lidar_shape_classifier/synthetic.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

namespace {

void print_help() {
  std::cout << "generate_scene --output scene.pcd [--seed N]\n\n"
            << "Write a synthetic room: ground, box, cylinder, sphere, vertical wall,\n"
            << "and a handful of outliers. The extension selects PCD or PLY.\n";
}

std::string require_value(int argc, char** argv, int& index, const char* flag) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + flag);
  }
  ++index;
  return argv[index];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string output;
    std::uint32_t seed = 1;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        print_help();
        return 0;
      }
      if (arg == "--output") {
        output = require_value(argc, argv, i, "--output");
      } else if (arg == "--seed") {
        const std::string value = require_value(argc, argv, i, "--seed");
        std::size_t used = 0;
        const unsigned long parsed = std::stoul(value, &used);
        if (used != value.size()) {
          throw std::runtime_error("invalid integer for --seed");
        }
        seed = static_cast<std::uint32_t>(parsed);
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }
    if (output.empty()) {
      print_help();
      throw std::runtime_error("--output is required");
    }
    const std::vector<Eigen::Vector3f> scene = lsc::make_demo_scene(seed);
    lsc::save_cloud(output, scene);
    std::cout << "wrote " << scene.size() << " points to " << output << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "generate_scene: " << ex.what() << "\n";
    return 1;
  }
}
