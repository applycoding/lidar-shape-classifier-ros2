#include "lidar_shape_classifier/pipeline.hpp"

#include "cloud_convert.hpp"

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

namespace lsc {

std::vector<Eigen::Vector3f> voxel_downsample(const std::vector<Eigen::Vector3f>& points,
                                              float leaf_size) {
  const std::vector<Eigen::Vector3f> finite = finite_points(points);
  if (finite.size() < 2 || leaf_size <= 0.f) {
    return finite;
  }
  const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = to_pcl(finite);
  pcl::VoxelGrid<pcl::PointXYZ> grid;
  grid.setInputCloud(cloud);
  grid.setLeafSize(leaf_size, leaf_size, leaf_size);
  pcl::PointCloud<pcl::PointXYZ> filtered;
  grid.filter(filtered);
  return from_pcl(filtered);
}

std::vector<Eigen::Vector3f> remove_statistical_outliers(const std::vector<Eigen::Vector3f>& points,
                                                         int mean_k, float stddev_mul) {
  const std::vector<Eigen::Vector3f> finite = finite_points(points);
  if (mean_k < 2 || stddev_mul <= 0.f || finite.size() <= static_cast<std::size_t>(mean_k)) {
    return finite;
  }
  const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = to_pcl(finite);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> filter;
  filter.setInputCloud(cloud);
  filter.setMeanK(mean_k);
  filter.setStddevMulThresh(stddev_mul);
  pcl::PointCloud<pcl::PointXYZ> filtered;
  filter.filter(filtered);
  return from_pcl(filtered);
}

}  // namespace lsc
