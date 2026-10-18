#include "lidar_shape_classifier/pipeline.hpp"

#include "cloud_convert.hpp"
#include "points.hpp"

#include <Eigen/Eigenvalues>

#include <pcl/console/print.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace lsc {
namespace {

constexpr float kPlaneInlierMin = 0.80f;
constexpr float kPlaneThinnessMax = 0.15f;
constexpr float kPlaneEigenRatioMax = 0.08f;
constexpr float kSphereSphericityMin = 0.58f;
constexpr float kSphereRadialCvMax = 0.08f;
constexpr float kSphereExtentRatioMax = 1.40f;
constexpr float kBoxFaceMin = 0.86f;
constexpr float kBoxThinnessMin = 0.22f;
constexpr float kBoxRadialCvMin = 0.10f;
constexpr float kCylinderInlierMin = 0.82f;
constexpr float kCylinderRadiusCvMax = 0.12f;
constexpr float kCylinderSectionRatioMax = 1.35f;
constexpr float kCylinderThinnessMin = 0.18f;

struct Pca {
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  Eigen::Vector3f axes[3] = {Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(),
                             Eigen::Vector3f::UnitZ()};
  Eigen::Vector3f extents = Eigen::Vector3f::Zero();
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  Eigen::Vector3f eigenvalues = Eigen::Vector3f::Zero();
  bool ok = false;
};

struct Radial {
  float mean = 0.f;
  float cv = 1.f;
};

float safe_div(float numerator, float denominator) {
  return numerator / std::max(denominator, 1e-8f);
}

Pca compute_pca(const std::vector<Eigen::Vector3f>& points) {
  Pca pca;
  if (points.size() < 3) {
    return pca;
  }
  Eigen::Vector3f sum = Eigen::Vector3f::Zero();
  for (const Eigen::Vector3f& point : points) {
    sum += point;
  }
  pca.centroid = sum / static_cast<float>(points.size());

  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  for (const Eigen::Vector3f& point : points) {
    const Eigen::Vector3f delta = point - pca.centroid;
    covariance += delta * delta.transpose();
  }
  covariance /= static_cast<float>(points.size());

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return pca;
  }

  pca.axes[0] = solver.eigenvectors().col(2).normalized();
  pca.axes[1] = solver.eigenvectors().col(1).normalized();
  pca.axes[2] = solver.eigenvectors().col(0).normalized();
  Eigen::Matrix3f rotation;
  rotation.col(0) = pca.axes[0];
  rotation.col(1) = pca.axes[1];
  rotation.col(2) = pca.axes[2];
  if (rotation.determinant() < 0.f) {
    pca.axes[1] = -pca.axes[1];
  }

  Eigen::Vector3f minimum = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
  Eigen::Vector3f maximum = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
  for (const Eigen::Vector3f& point : points) {
    const Eigen::Vector3f local(pca.axes[0].dot(point - pca.centroid),
                                pca.axes[1].dot(point - pca.centroid),
                                pca.axes[2].dot(point - pca.centroid));
    minimum = minimum.cwiseMin(local);
    maximum = maximum.cwiseMax(local);
  }
  const Eigen::Vector3f mid = 0.5f * (minimum + maximum);
  pca.center = pca.centroid + pca.axes[0] * mid.x() + pca.axes[1] * mid.y() + pca.axes[2] * mid.z();
  pca.extents = (maximum - minimum).cwiseMax(0.f);
  pca.eigenvalues = Eigen::Vector3f(solver.eigenvalues()(2), solver.eigenvalues()(1),
                                    solver.eigenvalues()(0));
  pca.ok = true;
  return pca;
}

Radial radial_stats(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& origin) {
  Radial stats;
  if (points.empty()) {
    return stats;
  }
  double sum = 0.0;
  for (const Eigen::Vector3f& point : points) {
    sum += (point - origin).norm();
  }
  stats.mean = static_cast<float>(sum / static_cast<double>(points.size()));
  double variance = 0.0;
  for (const Eigen::Vector3f& point : points) {
    const double delta = (point - origin).norm() - stats.mean;
    variance += delta * delta;
  }
  variance /= static_cast<double>(points.size());
  stats.cv = static_cast<float>(std::sqrt(std::max(0.0, variance))) / std::max(stats.mean, 1e-8f);
  return stats;
}

float score_plane(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& normal,
                  float offset, float threshold) {
  const float norm = normal.norm();
  if (norm < 1e-6f || points.empty()) {
    return 0.f;
  }
  const Eigen::Vector3f unit = normal / norm;
  const float d = offset / norm;
  int hits = 0;
  for (const Eigen::Vector3f& point : points) {
    if (std::abs(unit.dot(point) + d) <= threshold) {
      ++hits;
    }
  }
  return static_cast<float>(hits) / static_cast<float>(points.size());
}

float score_sphere(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& center,
                   float radius, float threshold) {
  if (radius <= 0.f || points.empty()) {
    return 0.f;
  }
  int hits = 0;
  for (const Eigen::Vector3f& point : points) {
    if (std::abs((point - center).norm() - radius) <= threshold) {
      ++hits;
    }
  }
  return static_cast<float>(hits) / static_cast<float>(points.size());
}

struct SphereFit {
  float inlier_ratio = 0.f;
  float radius = 0.f;
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
};

void silence_pcl() { pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS); }

float ransac_plane_score(const std::vector<Eigen::Vector3f>& points, float threshold, int iterations) {
  if (points.size() < 3) {
    return 0.f;
  }
  silence_pcl();
  const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = to_pcl(points);
  auto model = std::make_shared<pcl::SampleConsensusModelPlane<pcl::PointXYZ>>(cloud, false);
  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model);
  ransac.setDistanceThreshold(threshold);
  ransac.setMaxIterations(iterations);
  if (!ransac.computeModel()) {
    return 0.f;
  }
  Eigen::VectorXf coefficients;
  ransac.getModelCoefficients(coefficients);
  if (coefficients.size() < 4) {
    return 0.f;
  }
  return score_plane(points, coefficients.head<3>(), coefficients[3], threshold);
}

SphereFit ransac_sphere(const std::vector<Eigen::Vector3f>& points, float threshold, int iterations) {
  SphereFit fit;
  if (points.size() < 4) {
    return fit;
  }
  silence_pcl();
  const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = to_pcl(points);
  auto model = std::make_shared<pcl::SampleConsensusModelSphere<pcl::PointXYZ>>(cloud, false);
  model->setRadiusLimits(0.02, 5.0);
  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model);
  ransac.setDistanceThreshold(threshold);
  ransac.setMaxIterations(iterations);
  if (!ransac.computeModel()) {
    return fit;
  }
  Eigen::VectorXf coefficients;
  ransac.getModelCoefficients(coefficients);
  if (coefficients.size() < 4) {
    return fit;
  }
  fit.center = coefficients.head<3>();
  fit.radius = std::abs(coefficients[3]);
  fit.inlier_ratio = score_sphere(points, fit.center, fit.radius, threshold);
  return fit;
}

struct CylinderScore {
  float inlier_ratio = 0.f;
  float radius = 0.f;
  float radius_cv = 1.f;
  float section_ratio = 10.f;
  int axis_index = 0;
};

CylinderScore score_cylinder(const std::vector<Eigen::Vector3f>& points, const Pca& pca,
                             float threshold) {
  CylinderScore best;
  std::vector<float> distances(points.size());
  std::vector<float> scratch(points.size());
  for (int axis_index = 0; axis_index < 3; ++axis_index) {
    const Eigen::Vector3f& axis = pca.axes[axis_index];
    for (std::size_t i = 0; i < points.size(); ++i) {
      Eigen::Vector3f delta = points[i] - pca.center;
      delta -= axis * delta.dot(axis);
      distances[i] = delta.norm();
    }
    scratch = distances;
    const std::size_t mid = scratch.size() / 2;
    std::nth_element(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(mid),
                     scratch.end());
    const float radius = scratch[mid];
    if (radius < 0.02f) {
      continue;
    }
    int hits = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (const float distance : distances) {
      if (std::abs(distance - radius) <= threshold) {
        ++hits;
        sum += distance;
        sum_sq += static_cast<double>(distance) * distance;
      }
    }
    if (hits < 5) {
      continue;
    }
    const float ratio = static_cast<float>(hits) / static_cast<float>(points.size());
    const double mean = sum / static_cast<double>(hits);
    const double variance = sum_sq / static_cast<double>(hits) - mean * mean;
    const float cv = static_cast<float>(std::sqrt(std::max(0.0, variance))) / radius;
    const int a = (axis_index + 1) % 3;
    const int b = (axis_index + 2) % 3;
    const float section = safe_div(std::max(pca.extents[a], pca.extents[b]),
                                   std::min(pca.extents[a], pca.extents[b]));
    if (ratio > best.inlier_ratio) {
      best.inlier_ratio = ratio;
      best.radius = radius;
      best.radius_cv = cv;
      best.section_ratio = section;
      best.axis_index = axis_index;
    }
  }
  return best;
}

float score_box_faces(const std::vector<Eigen::Vector3f>& points, const Pca& pca, float threshold) {
  if (points.empty()) {
    return 0.f;
  }
  const Eigen::Vector3f half = 0.5f * pca.extents;
  int hits = 0;
  for (const Eigen::Vector3f& point : points) {
    const Eigen::Vector3f delta = point - pca.center;
    const float dx = half.x() - std::abs(pca.axes[0].dot(delta));
    const float dy = half.y() - std::abs(pca.axes[1].dot(delta));
    const float dz = half.z() - std::abs(pca.axes[2].dot(delta));
    if (std::min(dx, std::min(dy, dz)) <= threshold) {
      ++hits;
    }
  }
  return static_cast<float>(hits) / static_cast<float>(points.size());
}

ClassifiedCluster blank(const std::vector<Eigen::Vector3f>& points) {
  ClassifiedCluster cluster;
  cluster.point_count = points.size();
  if (points.empty()) {
    return cluster;
  }
  Eigen::Vector3f sum = Eigen::Vector3f::Zero();
  for (const Eigen::Vector3f& point : points) {
    sum += point;
  }
  cluster.centroid = sum / static_cast<float>(points.size());
  cluster.obb_center = cluster.centroid;
  return cluster;
}

void copy_frame(ClassifiedCluster& cluster, const Pca& pca) {
  cluster.centroid = pca.centroid;
  cluster.obb_center = pca.center;
  cluster.axis_x = pca.axes[0];
  cluster.axis_y = pca.axes[1];
  cluster.axis_z = pca.axes[2];
  cluster.obb_extents = pca.extents;
  cluster.axis = pca.axes[2];
}

}  // namespace

ClassifiedCluster classify_cluster(const std::vector<Eigen::Vector3f>& points,
                                   const PipelineConfig& config) {
  const std::vector<Eigen::Vector3f> finite = finite_points(points);
  ClassifiedCluster cluster = blank(finite);
  if (finite.size() < static_cast<std::size_t>(std::max(config.min_classify_points, 3))) {
    return cluster;
  }

  const Pca pca = compute_pca(finite);
  if (!pca.ok || pca.extents.maxCoeff() < 1e-4f || pca.eigenvalues[0] <= 1e-10f) {
    return cluster;
  }
  copy_frame(cluster, pca);

  const float threshold =
      config.fit_distance_threshold > 0.f ? config.fit_distance_threshold : 0.02f;
  const int iterations = std::max(config.fit_max_iterations, 20);
  const float max_extent = pca.extents.maxCoeff();
  const float min_extent = std::max(pca.extents.minCoeff(), 0.f);
  const float thinness = safe_div(min_extent, max_extent);
  const float extent_ratio = safe_div(max_extent, min_extent);
  const float sphericity = safe_div(pca.eigenvalues[2], pca.eigenvalues[0]);
  const Radial radial = radial_stats(finite, pca.centroid);

  const float pca_plane =
      score_plane(finite, pca.axes[2], -pca.axes[2].dot(pca.centroid), threshold);
  const float plane_ratio = std::max(pca_plane, ransac_plane_score(finite, threshold, iterations));
  const SphereFit sphere = ransac_sphere(finite, threshold, iterations);
  const CylinderScore cylinder = score_cylinder(finite, pca, threshold);
  const float face_ratio = score_box_faces(finite, pca, threshold);

  const bool plane_ok = plane_ratio >= kPlaneInlierMin && thinness <= kPlaneThinnessMax &&
                        sphericity <= kPlaneEigenRatioMax;
  const bool sphere_ok = sphericity >= kSphereSphericityMin && radial.cv <= kSphereRadialCvMax &&
                         extent_ratio <= kSphereExtentRatioMax;
  const bool box_ok = face_ratio >= kBoxFaceMin && thinness >= kBoxThinnessMin &&
                      radial.cv >= kBoxRadialCvMin && sphericity <= kSphereSphericityMin;
  const bool cylinder_ok = cylinder.inlier_ratio >= kCylinderInlierMin &&
                           cylinder.radius_cv <= kCylinderRadiusCvMax &&
                           cylinder.section_ratio <= kCylinderSectionRatioMax &&
                           thinness >= kCylinderThinnessMin && sphericity <= kSphereSphericityMin;

  if (plane_ok) {
    cluster.shape = ShapeClass::Plane;
    cluster.confidence = plane_ratio;
    cluster.axis = pca.axes[2];
    cluster.height = min_extent;
    return cluster;
  }
  if (sphere_ok) {
    cluster.shape = ShapeClass::Sphere;
    const bool ransac_agrees =
        sphere.inlier_ratio >= 0.60f && sphere.radius > 0.f &&
        std::abs(sphere.radius - radial.mean) <= 0.25f * std::max(radial.mean, 1e-4f);
    cluster.radius = ransac_agrees ? sphere.radius : radial.mean;
    cluster.confidence = std::max(sphere.inlier_ratio, 1.f - radial.cv);
    cluster.axis = Eigen::Vector3f::UnitZ();
    return cluster;
  }
  if (box_ok && (!cylinder_ok || face_ratio >= cylinder.inlier_ratio)) {
    cluster.shape = ShapeClass::Box;
    cluster.confidence = face_ratio;
    return cluster;
  }
  if (cylinder_ok) {
    cluster.shape = ShapeClass::Cylinder;
    cluster.confidence = cylinder.inlier_ratio;
    cluster.radius = cylinder.radius;
    cluster.axis = pca.axes[cylinder.axis_index];
    cluster.height = pca.extents[cylinder.axis_index];
    return cluster;
  }
  return cluster;
}

}  // namespace lsc
