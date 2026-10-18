#include "lidar_shape_classifier/pipeline.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/quaternion.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace {

struct Rgb {
  float r;
  float g;
  float b;
};

Rgb color_for(lsc::ShapeClass shape) {
  switch (shape) {
    case lsc::ShapeClass::Box:
      return {0.93f, 0.55f, 0.18f};
    case lsc::ShapeClass::Cylinder:
      return {0.20f, 0.45f, 0.92f};
    case lsc::ShapeClass::Sphere:
      return {0.18f, 0.72f, 0.38f};
    case lsc::ShapeClass::Plane:
      return {0.90f, 0.82f, 0.20f};
    case lsc::ShapeClass::Unknown:
      return {0.55f, 0.55f, 0.58f};
  }
  throw std::logic_error("unhandled ShapeClass");
}

float display_scale(float value) { return std::max(value, 0.01f); }

void set_quaternion(geometry_msgs::msg::Quaternion& out, const Eigen::Quaternionf& rotation) {
  const Eigen::Quaternionf q = rotation.normalized();
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
}

visualization_msgs::msg::Marker shape_marker(const lsc::ClassifiedCluster& cluster,
                                              const std_msgs::msg::Header& header, int id) {
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.ns = "shapes";
  marker.id = id;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.position.x = cluster.obb_center.x();
  marker.pose.position.y = cluster.obb_center.y();
  marker.pose.position.z = cluster.obb_center.z();
  const Rgb rgb = color_for(cluster.shape);
  marker.color.r = rgb.r;
  marker.color.g = rgb.g;
  marker.color.b = rgb.b;
  marker.color.a = 0.45f;

  switch (cluster.shape) {
    case lsc::ShapeClass::Sphere: {
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      const float diameter = display_scale(2.f * cluster.radius);
      marker.scale.x = diameter;
      marker.scale.y = diameter;
      marker.scale.z = diameter;
      marker.pose.orientation.w = 1.f;
      break;
    }
    case lsc::ShapeClass::Cylinder: {
      marker.type = visualization_msgs::msg::Marker::CYLINDER;
      const float diameter = display_scale(2.f * cluster.radius);
      marker.scale.x = diameter;
      marker.scale.y = diameter;
      marker.scale.z = display_scale(cluster.height);
      const Eigen::Vector3f axis = cluster.axis.norm() > 1e-6f ? cluster.axis.normalized()
                                                               : Eigen::Vector3f::UnitZ();
      set_quaternion(marker.pose.orientation,
                     Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), axis));
      break;
    }
    case lsc::ShapeClass::Box:
    case lsc::ShapeClass::Plane:
    case lsc::ShapeClass::Unknown: {
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.scale.x = display_scale(cluster.obb_extents.x());
      marker.scale.y = display_scale(cluster.obb_extents.y());
      marker.scale.z = display_scale(cluster.obb_extents.z());
      Eigen::Matrix3f rotation;
      rotation.col(0) = cluster.axis_x;
      rotation.col(1) = cluster.axis_y;
      rotation.col(2) = cluster.axis_z;
      if (rotation.determinant() < 0.f) {
        rotation.col(2) *= -1.f;
      }
      set_quaternion(marker.pose.orientation, Eigen::Quaternionf(rotation));
      break;
    }
  }
  return marker;
}

visualization_msgs::msg::Marker label_marker(const lsc::ClassifiedCluster& cluster,
                                              const std_msgs::msg::Header& header, int id) {
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.ns = "labels";
  marker.id = id;
  marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.text = std::string(lsc::to_string(cluster.shape));
  marker.scale.z = 0.15f;
  marker.color.r = 1.f;
  marker.color.g = 1.f;
  marker.color.b = 1.f;
  marker.color.a = 1.f;
  marker.pose.orientation.w = 1.f;
  marker.pose.position.x = cluster.centroid.x();
  marker.pose.position.y = cluster.centroid.y();
  marker.pose.position.z =
      cluster.centroid.z() + 0.5f * cluster.obb_extents.maxCoeff() + 0.12f;
  return marker;
}

class ShapeClassifierNode : public rclcpp::Node {
 public:
  ShapeClassifierNode() : rclcpp::Node("shape_classifier") {
    config_.voxel_leaf_size = static_cast<float>(
        declare_parameter<double>("voxel_leaf_size", config_.voxel_leaf_size));
    config_.outlier_mean_k = declare_parameter<int>("outlier_mean_k", config_.outlier_mean_k);
    config_.outlier_stddev_mul = static_cast<float>(
        declare_parameter<double>("outlier_stddev_mul", config_.outlier_stddev_mul));
    config_.ground_distance_threshold = static_cast<float>(
        declare_parameter<double>("ground_distance_threshold", config_.ground_distance_threshold));
    config_.ground_max_iterations =
        declare_parameter<int>("ground_max_iterations", config_.ground_max_iterations);
    config_.ground_eps_angle_deg = static_cast<float>(
        declare_parameter<double>("ground_eps_angle_deg", config_.ground_eps_angle_deg));
    config_.cluster_tolerance = static_cast<float>(
        declare_parameter<double>("cluster_tolerance", config_.cluster_tolerance));
    config_.cluster_min_size = declare_parameter<int>("cluster_min_size", config_.cluster_min_size);
    config_.cluster_max_size = declare_parameter<int>("cluster_max_size", config_.cluster_max_size);
    config_.fit_distance_threshold = static_cast<float>(
        declare_parameter<double>("fit_distance_threshold", config_.fit_distance_threshold));
    config_.fit_max_iterations =
        declare_parameter<int>("fit_max_iterations", config_.fit_max_iterations);
    config_.min_classify_points =
        declare_parameter<int>("min_classify_points", config_.min_classify_points);

    const std::string input_topic = declare_parameter<std::string>("input_topic", "/points");
    const std::string marker_topic =
        declare_parameter<std::string>("marker_topic", "/shape_markers");
    const std::string non_ground_topic =
        declare_parameter<std::string>("non_ground_topic", "/shape_non_ground");
    frame_id_ = declare_parameter<std::string>("frame_id", "lidar");

    markers_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic, rclcpp::QoS(10));
    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(non_ground_topic,
                                                                 rclcpp::SensorDataQoS());
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic, rclcpp::SensorDataQoS(),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { on_cloud(msg); });
  }

 private:
  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
    pcl::PointCloud<pcl::PointXYZ> input;
    pcl::fromROSMsg(*msg, input);
    std::vector<Eigen::Vector3f> points;
    points.reserve(input.size());
    for (const pcl::PointXYZ& point : input.points) {
      points.emplace_back(point.x, point.y, point.z);
    }

    const lsc::PipelineResult result = lsc::run_pipeline(points, config_);

    std_msgs::msg::Header header = msg->header;
    if (header.frame_id.empty()) {
      header.frame_id = frame_id_;
    }

    visualization_msgs::msg::MarkerArray clear;
    visualization_msgs::msg::Marker wipe;
    wipe.header = header;
    wipe.action = visualization_msgs::msg::Marker::DELETEALL;
    clear.markers.push_back(wipe);
    markers_pub_->publish(clear);

    visualization_msgs::msg::MarkerArray markers;
    markers.markers.reserve(result.clusters.size() * 2);
    for (std::size_t i = 0; i < result.clusters.size(); ++i) {
      const int id = static_cast<int>(i);
      markers.markers.push_back(shape_marker(result.clusters[i], header, id));
      markers.markers.push_back(label_marker(result.clusters[i], header, id));
    }
    markers_pub_->publish(markers);

    pcl::PointCloud<pcl::PointXYZ> non_ground;
    non_ground.reserve(result.non_ground.size());
    for (const Eigen::Vector3f& point : result.non_ground) {
      non_ground.push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }
    non_ground.width = static_cast<std::uint32_t>(non_ground.size());
    non_ground.height = 1;
    non_ground.is_dense = true;
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(non_ground, cloud_msg);
    cloud_msg.header = header;
    cloud_pub_->publish(cloud_msg);

    RCLCPP_INFO_STREAM_THROTTLE(get_logger(), *get_clock(), 2000,
                                "classified " << result.clusters.size() << " clusters from "
                                              << result.input_points << " points");
  }

  lsc::PipelineConfig config_;
  std::string frame_id_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ShapeClassifierNode>());
  rclcpp::shutdown();
  return 0;
}
