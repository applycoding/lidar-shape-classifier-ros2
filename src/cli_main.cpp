#include "lidar_shape_classifier/pipeline.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void print_help() {
  const lsc::PipelineConfig defaults;
  std::cout << "shape_classify --input cloud.pcd [options]\n\n"
            << "Run the geometric shape pipeline on a PCD or PLY file.\n"
            << "A non-positive --leaf or --outlier-stddev skips that stage.\n\n"
            << "  --input PATH             input cloud\n"
            << "  --leaf METERS            voxel leaf (" << defaults.voxel_leaf_size << ")\n"
            << "  --outlier-k N            neighbor count (" << defaults.outlier_mean_k << ")\n"
            << "  --outlier-stddev X       stddev multiplier (" << defaults.outlier_stddev_mul << ")\n"
            << "  --ground-threshold X     ground RANSAC distance ("
            << defaults.ground_distance_threshold << ")\n"
            << "  --cluster-tolerance X    Euclidean tolerance (" << defaults.cluster_tolerance
            << ")\n"
            << "  --cluster-min N          minimum cluster size (" << defaults.cluster_min_size
            << ")\n"
            << "  --fit-threshold X        model inlier distance (" << defaults.fit_distance_threshold
            << ")\n"
            << "  --help                   show this text\n";
}

float parse_float(const std::string& value, const char* flag) {
  try {
    std::size_t used = 0;
    const float parsed = std::stof(value, &used);
    if (used != value.size()) {
      throw std::runtime_error("trailing junk");
    }
    return parsed;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid number for ") + flag);
  }
}

int parse_int(const std::string& value, const char* flag) {
  try {
    std::size_t used = 0;
    const int parsed = std::stoi(value, &used);
    if (used != value.size()) {
      throw std::runtime_error("trailing junk");
    }
    return parsed;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid integer for ") + flag);
  }
}

std::string require_value(int argc, char** argv, int& index, const char* flag) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + flag);
  }
  ++index;
  return argv[index];
}

void print_cluster(int index, const lsc::ClassifiedCluster& cluster) {
  std::cout << "  [" << index << "] " << std::left << std::setw(9) << lsc::to_string(cluster.shape)
            << std::right << " points=" << std::setw(5) << cluster.point_count
            << "  conf=" << std::fixed << std::setprecision(3) << cluster.confidence
            << "  centroid=(" << cluster.centroid.x() << ", " << cluster.centroid.y() << ", "
            << cluster.centroid.z() << ")";
  if (cluster.shape == lsc::ShapeClass::Sphere || cluster.shape == lsc::ShapeClass::Cylinder) {
    std::cout << "  radius=" << cluster.radius;
  }
  if (cluster.shape == lsc::ShapeClass::Cylinder) {
    std::cout << "  height=" << cluster.height;
  }
  if (cluster.shape == lsc::ShapeClass::Box || cluster.shape == lsc::ShapeClass::Plane ||
      cluster.shape == lsc::ShapeClass::Unknown) {
    std::cout << "  extents=(" << cluster.obb_extents.x() << ", " << cluster.obb_extents.y() << ", "
              << cluster.obb_extents.z() << ")";
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string input;
    lsc::PipelineConfig config;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        print_help();
        return 0;
      }
      if (arg == "--input") {
        input = require_value(argc, argv, i, "--input");
      } else if (arg == "--leaf") {
        config.voxel_leaf_size = parse_float(require_value(argc, argv, i, "--leaf"), "--leaf");
      } else if (arg == "--outlier-k") {
        config.outlier_mean_k = parse_int(require_value(argc, argv, i, "--outlier-k"), "--outlier-k");
      } else if (arg == "--outlier-stddev") {
        config.outlier_stddev_mul =
            parse_float(require_value(argc, argv, i, "--outlier-stddev"), "--outlier-stddev");
      } else if (arg == "--ground-threshold") {
        config.ground_distance_threshold = parse_float(
            require_value(argc, argv, i, "--ground-threshold"), "--ground-threshold");
      } else if (arg == "--cluster-tolerance") {
        config.cluster_tolerance = parse_float(
            require_value(argc, argv, i, "--cluster-tolerance"), "--cluster-tolerance");
      } else if (arg == "--cluster-min") {
        config.cluster_min_size =
            parse_int(require_value(argc, argv, i, "--cluster-min"), "--cluster-min");
      } else if (arg == "--fit-threshold") {
        config.fit_distance_threshold =
            parse_float(require_value(argc, argv, i, "--fit-threshold"), "--fit-threshold");
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }
    if (input.empty()) {
      print_help();
      throw std::runtime_error("--input is required");
    }

    const std::vector<Eigen::Vector3f> points = lsc::load_cloud(input);
    const lsc::PipelineResult result = lsc::run_pipeline(points, config);
    std::cout << "input: " << result.input_points << "\n"
              << "after_voxel: " << result.voxel_points << "\n"
              << "after_outliers: " << result.inlier_points << "\n"
              << "after_ground: " << result.non_ground_points << "\n"
              << "ground_removed: " << (result.ground_removed ? "yes" : "no") << "\n"
              << "clusters: " << result.clusters.size() << "\n";
    for (std::size_t i = 0; i < result.clusters.size(); ++i) {
      print_cluster(static_cast<int>(i), result.clusters[i]);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "shape_classify: " << ex.what() << "\n";
    return 1;
  }
}
