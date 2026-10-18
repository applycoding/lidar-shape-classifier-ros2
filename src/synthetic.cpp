#include "lidar_shape_classifier/synthetic.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace lsc {
namespace {

constexpr float kPi = 3.14159265358979323846f;

std::mt19937 make_rng(std::uint32_t seed) { return std::mt19937{seed}; }

Eigen::Vector3f jitter(std::mt19937& rng, float noise) {
  if (noise <= 0.f) {
    return Eigen::Vector3f::Zero();
  }
  std::normal_distribution<float> gauss(0.f, noise);
  return Eigen::Vector3f(gauss(rng), gauss(rng), gauss(rng));
}

void plane_basis(const Eigen::Vector3f& normal, Eigen::Vector3f& tangent_u, Eigen::Vector3f& tangent_v) {
  const Eigen::Vector3f unit = normal.normalized();
  const Eigen::Vector3f helper =
      std::abs(unit.z()) < 0.9f ? Eigen::Vector3f::UnitZ() : Eigen::Vector3f::UnitX();
  tangent_u = unit.cross(helper).normalized();
  tangent_v = unit.cross(tangent_u).normalized();
}

float unit_coord(int index, int count) {
  if (count <= 1) {
    return 0.f;
  }
  return static_cast<float>(index) / static_cast<float>(count - 1) - 0.5f;
}

}  // namespace

std::vector<Eigen::Vector3f> make_sphere(const Eigen::Vector3f& center, float radius, int count,
                                         float noise, std::uint32_t seed) {
  std::vector<Eigen::Vector3f> points;
  if (count <= 0 || radius == 0.f) {
    return points;
  }
  points.reserve(static_cast<std::size_t>(count));
  std::mt19937 rng = make_rng(seed);
  const float golden = kPi * (3.f - std::sqrt(5.f));
  const float abs_radius = std::abs(radius);
  for (int i = 0; i < count; ++i) {
    const float y = 1.f - (2.f * static_cast<float>(i)) / static_cast<float>(std::max(count - 1, 1));
    const float ring = std::sqrt(std::max(0.f, 1.f - y * y));
    const float theta = golden * static_cast<float>(i);
    Eigen::Vector3f point(ring * std::cos(theta), y, ring * std::sin(theta));
    point = center + abs_radius * point + jitter(rng, noise);
    points.push_back(point);
  }
  return points;
}

std::vector<Eigen::Vector3f> make_box(const Eigen::Vector3f& center, const Eigen::Vector3f& size,
                                      int points_along_edge, float noise, std::uint32_t seed,
                                      float yaw_rad) {
  std::vector<Eigen::Vector3f> points;
  const int samples = std::max(points_along_edge, 2);
  points.reserve(static_cast<std::size_t>(samples) * static_cast<std::size_t>(samples) * 6u);
  std::mt19937 rng = make_rng(seed);
  const Eigen::Matrix3f rotation =
      Eigen::AngleAxisf(yaw_rad, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  const Eigen::Vector3f extent = size.cwiseAbs();
  for (int fixed = 0; fixed < 3; ++fixed) {
    for (const int sign : {1, -1}) {
      for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < samples; ++j) {
          Eigen::Vector3f local = Eigen::Vector3f::Zero();
          const int axis_a = (fixed + 1) % 3;
          const int axis_b = (fixed + 2) % 3;
          local[fixed] = static_cast<float>(sign) * extent[fixed] * 0.5f;
          local[axis_a] = unit_coord(i, samples) * extent[axis_a];
          local[axis_b] = unit_coord(j, samples) * extent[axis_b];
          points.push_back(center + rotation * local + jitter(rng, noise));
        }
      }
    }
  }
  return points;
}

std::vector<Eigen::Vector3f> make_cylinder(const Eigen::Vector3f& center, float radius, float height,
                                           int n_theta, int n_height, float noise,
                                           std::uint32_t seed) {
  std::vector<Eigen::Vector3f> points;
  const int theta_count = std::max(n_theta, 3);
  const int height_count = std::max(n_height, 2);
  points.reserve(static_cast<std::size_t>(theta_count) * static_cast<std::size_t>(height_count));
  std::mt19937 rng = make_rng(seed);
  const float abs_radius = std::abs(radius);
  const float abs_height = std::abs(height);
  for (int iz = 0; iz < height_count; ++iz) {
    const float z = unit_coord(iz, height_count) * abs_height;
    for (int it = 0; it < theta_count; ++it) {
      const float theta = 2.f * kPi * static_cast<float>(it) / static_cast<float>(theta_count);
      Eigen::Vector3f point(abs_radius * std::cos(theta), abs_radius * std::sin(theta), z);
      points.push_back(center + point + jitter(rng, noise));
    }
  }
  return points;
}

std::vector<Eigen::Vector3f> make_plane(const Eigen::Vector3f& center, const Eigen::Vector3f& normal,
                                        float width, float height, int nu, int nv, float noise,
                                        std::uint32_t seed) {
  std::vector<Eigen::Vector3f> points;
  if (normal.norm() < 1e-6f) {
    return points;
  }
  const int u_count = std::max(nu, 2);
  const int v_count = std::max(nv, 2);
  points.reserve(static_cast<std::size_t>(u_count) * static_cast<std::size_t>(v_count));
  std::mt19937 rng = make_rng(seed);
  Eigen::Vector3f tangent_u;
  Eigen::Vector3f tangent_v;
  plane_basis(normal, tangent_u, tangent_v);
  for (int iv = 0; iv < v_count; ++iv) {
    for (int iu = 0; iu < u_count; ++iu) {
      const Eigen::Vector3f point = center + unit_coord(iu, u_count) * std::abs(width) * tangent_u +
                                    unit_coord(iv, v_count) * std::abs(height) * tangent_v +
                                    jitter(rng, noise);
      points.push_back(point);
    }
  }
  return points;
}

std::vector<Eigen::Vector3f> make_demo_scene(std::uint32_t seed) {
  // Room-scale scene in meters.
  // Ground at z = 0. Objects sit with their bottoms at z = 0.25 so ground
  // removal and clustering stay separate. A vertical wall is the plane class.
  std::vector<Eigen::Vector3f> points;
  const auto append = [&points](const std::vector<Eigen::Vector3f>& extra) {
    points.insert(points.end(), extra.begin(), extra.end());
  };
  append(make_plane(Eigen::Vector3f::Zero(), Eigen::Vector3f::UnitZ(), 4.f, 4.f, 80, 80, 0.002f,
                    seed));
  append(make_box(Eigen::Vector3f(1.40f, 0.10f, 0.45f), Eigen::Vector3f(0.70f, 0.48f, 0.40f), 16,
                  0.001f, seed + 1u, 0.35f));
  append(make_cylinder(Eigen::Vector3f(-1.30f, 0.90f, 0.80f), 0.30f, 1.10f, 48, 28, 0.001f,
                       seed + 2u));
  append(make_sphere(Eigen::Vector3f(0.15f, -1.45f, 0.65f), 0.40f, 900, 0.001f, seed + 3u));
  append(make_plane(Eigen::Vector3f(0.f, 1.90f, 0.80f), Eigen::Vector3f::UnitY(), 1.40f, 1.10f, 36,
                    28, 0.001f, seed + 4u));

  std::mt19937 rng = make_rng(seed + 5u);
  std::uniform_real_distribution<float> spread(-6.f, 6.f);
  for (int i = 0; i < 30; ++i) {
    points.emplace_back(spread(rng), spread(rng), spread(rng));
  }
  return points;
}

}  // namespace lsc
