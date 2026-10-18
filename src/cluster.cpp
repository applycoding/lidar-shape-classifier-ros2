#include "lidar_shape_classifier/pipeline.hpp"

#include "cloud_convert.hpp"

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace lsc {

std::vector<std::vector<Eigen::Vector3f>> euclidean_clusters(
    const std::vector<Eigen::Vector3f>& points, const PipelineConfig& config) {
  std::vector<std::vector<Eigen::Vector3f>> clusters;
  const std::vector<Eigen::Vector3f> finite = finite_points(points);
  if (config.cluster_tolerance <= 0.f || config.cluster_min_size < 1 ||
      finite.size() < static_cast<std::size_t>(config.cluster_min_size)) {
    return clusters;
  }

  const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = to_pcl(finite);
  auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
  tree->setInputCloud(cloud);

  pcl::EuclideanClusterExtraction<pcl::PointXYZ> extractor;
  extractor.setClusterTolerance(config.cluster_tolerance);
  extractor.setMinClusterSize(config.cluster_min_size);
  extractor.setMaxClusterSize(std::max(config.cluster_max_size, config.cluster_min_size));
  extractor.setSearchMethod(tree);
  extractor.setInputCloud(cloud);

  std::vector<pcl::PointIndices> indices;
  extractor.extract(indices);
  clusters.reserve(indices.size());
  for (const pcl::PointIndices& cluster : indices) {
    std::vector<Eigen::Vector3f> group;
    group.reserve(cluster.indices.size());
    for (const int index : cluster.indices) {
      if (index >= 0 && static_cast<std::size_t>(index) < finite.size()) {
        group.push_back(finite[static_cast<std::size_t>(index)]);
      }
    }
    if (!group.empty()) {
      clusters.push_back(std::move(group));
    }
  }
  return clusters;
}

}  // namespace lsc
