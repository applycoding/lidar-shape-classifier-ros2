#include "lidar_shape_classifier/pipeline.hpp"
#include "lidar_shape_classifier/synthetic.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

lsc::ClassifiedCluster classify(const std::vector<Eigen::Vector3f>& points) {
  return lsc::classify_cluster(points, lsc::PipelineConfig{});
}

std::string scene_debug(const lsc::PipelineResult& result) {
  std::ostringstream out;
  out << "ground_removed=" << result.ground_removed << " voxel=" << result.voxel_points
      << " inliers=" << result.inlier_points << " non_ground=" << result.non_ground_points
      << "\n";
  for (const lsc::ClassifiedCluster& cluster : result.clusters) {
    out << lsc::to_string(cluster.shape) << " n=" << cluster.point_count
        << " conf=" << cluster.confidence << " radius=" << cluster.radius
        << " height=" << cluster.height << "\n";
  }
  return out.str();
}

}  // namespace

TEST(Preprocess, VoxelReducesDenseGrid) {
  std::vector<Eigen::Vector3f> points;
  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 5; ++k) {
        points.emplace_back(0.01f * static_cast<float>(i), 0.01f * static_cast<float>(j),
                            0.01f * static_cast<float>(k));
      }
    }
  }
  const std::vector<Eigen::Vector3f> down = lsc::voxel_downsample(points, 0.05f);
  EXPECT_LT(down.size(), points.size() / 4);
  EXPECT_GT(down.size(), 8u);
}

TEST(Preprocess, NonPositiveLeafKeepsPoints) {
  const std::vector<Eigen::Vector3f> points = lsc::make_sphere(Eigen::Vector3f::Zero(), 0.2f, 40, 0.f, 1);
  EXPECT_EQ(lsc::voxel_downsample(points, 0.f).size(), points.size());
}

TEST(Preprocess, StatisticalOutliersRemoved) {
  std::vector<Eigen::Vector3f> points;
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      for (int k = 0; k < 3; ++k) {
        points.emplace_back(0.02f * static_cast<float>(i), 0.02f * static_cast<float>(j),
                            0.02f * static_cast<float>(k));
      }
    }
  }
  const std::size_t core = points.size();
  for (int n = 0; n < 6; ++n) {
    points.emplace_back(10.f + static_cast<float>(n), -8.f, 4.f);
  }
  const std::vector<Eigen::Vector3f> filtered = lsc::remove_statistical_outliers(points, 12, 1.0f);
  EXPECT_LT(filtered.size(), points.size());
  EXPECT_GE(filtered.size(), core - 30);
  for (const Eigen::Vector3f& point : filtered) {
    EXPECT_LT(point.x(), 2.f);
  }
}

TEST(Ground, RemovesHorizontalPlane) {
  const std::vector<Eigen::Vector3f> ground =
      lsc::make_plane(Eigen::Vector3f::Zero(), Eigen::Vector3f::UnitZ(), 2.f, 2.f, 40, 40, 0.001f, 7);
  const std::vector<Eigen::Vector3f> box =
      lsc::make_box(Eigen::Vector3f(0.f, 0.f, 0.8f), Eigen::Vector3f(0.3f, 0.3f, 0.3f), 8, 0.001f, 8);
  std::vector<Eigen::Vector3f> points = ground;
  points.insert(points.end(), box.begin(), box.end());

  const lsc::GroundRemoval removed = lsc::remove_ground_plane(points, lsc::PipelineConfig{});
  EXPECT_TRUE(removed.removed);
  EXPECT_GT(removed.non_ground.size(), box.size() / 2);
  EXPECT_LT(removed.non_ground.size(), box.size() + 20);
  for (const Eigen::Vector3f& point : removed.non_ground) {
    EXPECT_GT(point.z(), 0.4f);
  }
}

TEST(Cluster, SplitsDistantGroups) {
  std::vector<Eigen::Vector3f> points =
      lsc::make_sphere(Eigen::Vector3f::Zero(), 0.2f, 80, 0.f, 1);
  const std::vector<Eigen::Vector3f> other =
      lsc::make_sphere(Eigen::Vector3f(3.f, 0.f, 0.f), 0.2f, 90, 0.f, 2);
  points.insert(points.end(), other.begin(), other.end());

  lsc::PipelineConfig config;
  config.cluster_min_size = 20;
  config.cluster_tolerance = 0.25f;
  const std::vector<std::vector<Eigen::Vector3f>> clusters = lsc::euclidean_clusters(points, config);
  ASSERT_EQ(clusters.size(), 2u);
  EXPECT_GT(clusters[0].size(), 40u);
  EXPECT_GT(clusters[1].size(), 40u);
}

TEST(Classify, Sphere) {
  const std::vector<Eigen::Vector3f> points =
      lsc::make_sphere(Eigen::Vector3f(0.2f, -0.3f, 0.5f), 0.4f, 500, 0.001f, 3);
  const lsc::ClassifiedCluster result = classify(points);
  EXPECT_EQ(result.shape, lsc::ShapeClass::Sphere) << lsc::to_string(result.shape);
  EXPECT_GT(result.confidence, 0.8f);
  EXPECT_NEAR(result.radius, 0.4f, 0.03f);
}

TEST(Classify, Box) {
  const std::vector<Eigen::Vector3f> points = lsc::make_box(
      Eigen::Vector3f(0.4f, 0.1f, 0.2f), Eigen::Vector3f(0.70f, 0.48f, 0.40f), 12, 0.001f, 4);
  const lsc::ClassifiedCluster result = classify(points);
  EXPECT_EQ(result.shape, lsc::ShapeClass::Box) << lsc::to_string(result.shape);
  EXPECT_GT(result.confidence, 0.8f);
}

TEST(Classify, RotatedBox) {
  const std::vector<Eigen::Vector3f> points =
      lsc::make_box(Eigen::Vector3f::Zero(), Eigen::Vector3f(0.70f, 0.48f, 0.40f), 12, 0.001f, 5, 0.6f);
  const lsc::ClassifiedCluster result = classify(points);
  EXPECT_EQ(result.shape, lsc::ShapeClass::Box) << lsc::to_string(result.shape);
  EXPECT_GT(result.confidence, 0.8f);
}

TEST(Classify, Cylinder) {
  const std::vector<Eigen::Vector3f> points =
      lsc::make_cylinder(Eigen::Vector3f(0.f, 0.2f, 0.5f), 0.30f, 1.10f, 40, 24, 0.001f, 6);
  const lsc::ClassifiedCluster result = classify(points);
  EXPECT_EQ(result.shape, lsc::ShapeClass::Cylinder) << lsc::to_string(result.shape);
  EXPECT_GT(result.confidence, 0.8f);
  EXPECT_NEAR(result.radius, 0.30f, 0.03f);
  EXPECT_NEAR(result.height, 1.10f, 0.08f);
}

TEST(Classify, Plane) {
  const std::vector<Eigen::Vector3f> points = lsc::make_plane(
      Eigen::Vector3f(0.f, 1.f, 0.4f), Eigen::Vector3f::UnitY(), 1.2f, 0.9f, 24, 18, 0.001f, 9);
  const lsc::ClassifiedCluster result = classify(points);
  EXPECT_EQ(result.shape, lsc::ShapeClass::Plane) << lsc::to_string(result.shape);
  EXPECT_GT(result.confidence, 0.8f);
}

TEST(Classify, BlobIsUnknown) {
  std::mt19937 rng(4);
  std::normal_distribution<float> gauss(0.f, 0.25f);
  std::vector<Eigen::Vector3f> points;
  points.reserve(200);
  for (int i = 0; i < 200; ++i) {
    points.emplace_back(gauss(rng), gauss(rng), gauss(rng));
  }
  const lsc::ClassifiedCluster result = classify(points);
  EXPECT_EQ(result.shape, lsc::ShapeClass::Unknown) << lsc::to_string(result.shape);
  EXPECT_FLOAT_EQ(result.confidence, 0.f);
}

TEST(Pipeline, DemoSceneLabels) {
  const lsc::PipelineResult result = lsc::run_pipeline(lsc::make_demo_scene(1));
  const std::string debug = scene_debug(result);
  EXPECT_TRUE(result.ground_removed) << debug;
  std::map<lsc::ShapeClass, int> counts;
  for (const lsc::ClassifiedCluster& cluster : result.clusters) {
    ++counts[cluster.shape];
  }
  EXPECT_EQ(counts[lsc::ShapeClass::Box], 1) << debug;
  EXPECT_EQ(counts[lsc::ShapeClass::Cylinder], 1) << debug;
  EXPECT_EQ(counts[lsc::ShapeClass::Sphere], 1) << debug;
  EXPECT_EQ(counts[lsc::ShapeClass::Plane], 1) << debug;
  EXPECT_EQ(counts[lsc::ShapeClass::Unknown], 0) << debug;
  EXPECT_EQ(result.clusters.size(), 4u) << debug;
}

TEST(Pipeline, EmptyCloud) {
  const lsc::PipelineResult result = lsc::run_pipeline({});
  EXPECT_EQ(result.input_points, 0u);
  EXPECT_TRUE(result.clusters.empty());
  EXPECT_FALSE(result.ground_removed);
}

TEST(Io, RoundTripPcdAndPly) {
  const std::vector<Eigen::Vector3f> points =
      lsc::make_box(Eigen::Vector3f(0.1f, 0.2f, 0.3f), Eigen::Vector3f(0.4f, 0.3f, 0.2f), 4, 0.f, 2);
  const std::string root(testing::TempDir());
  for (const char* name : {"lsc_roundtrip.pcd", "lsc_roundtrip.ply"}) {
    const std::string path = root + name;
    lsc::save_cloud(path, points);
    const std::vector<Eigen::Vector3f> loaded = lsc::load_cloud(path);
    ASSERT_EQ(loaded.size(), points.size()) << path;
    for (std::size_t i = 0; i < points.size(); ++i) {
      EXPECT_NEAR(loaded[i].x(), points[i].x(), 1e-4f) << path;
      EXPECT_NEAR(loaded[i].y(), points[i].y(), 1e-4f) << path;
      EXPECT_NEAR(loaded[i].z(), points[i].z(), 1e-4f) << path;
    }
  }
}

TEST(Io, MissingFileThrows) {
  EXPECT_THROW(lsc::load_cloud("/tmp/lsc_missing_cloud.pcd"), std::runtime_error);
  EXPECT_THROW(lsc::load_cloud("/tmp/lsc_missing_cloud.xyz"), std::runtime_error);
}

TEST(Types, ShapeNames) {
  EXPECT_EQ(lsc::to_string(lsc::ShapeClass::Box), "box");
  EXPECT_EQ(lsc::to_string(lsc::ShapeClass::Cylinder), "cylinder");
  EXPECT_EQ(lsc::to_string(lsc::ShapeClass::Sphere), "sphere");
  EXPECT_EQ(lsc::to_string(lsc::ShapeClass::Plane), "plane");
  EXPECT_EQ(lsc::to_string(lsc::ShapeClass::Unknown), "unknown");
}
