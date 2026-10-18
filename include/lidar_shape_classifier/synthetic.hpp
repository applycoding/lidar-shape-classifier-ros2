#pragma once

#include "lidar_shape_classifier/types.hpp"

#include <cstdint>
#include <vector>

namespace lsc {

std::vector<Eigen::Vector3f> make_sphere(const Eigen::Vector3f& center, float radius, int count,
                                         float noise, std::uint32_t seed);

std::vector<Eigen::Vector3f> make_box(const Eigen::Vector3f& center, const Eigen::Vector3f& size,
                                      int points_along_edge, float noise, std::uint32_t seed,
                                      float yaw_rad = 0.f);

std::vector<Eigen::Vector3f> make_cylinder(const Eigen::Vector3f& center, float radius, float height,
                                           int n_theta, int n_height, float noise,
                                           std::uint32_t seed);

std::vector<Eigen::Vector3f> make_plane(const Eigen::Vector3f& center, const Eigen::Vector3f& normal,
                                        float width, float height, int nu, int nv, float noise,
                                        std::uint32_t seed);

// Ground, a box, a cylinder, a sphere, a vertical wall, and a few outliers.
std::vector<Eigen::Vector3f> make_demo_scene(std::uint32_t seed = 1);

}  // namespace lsc
