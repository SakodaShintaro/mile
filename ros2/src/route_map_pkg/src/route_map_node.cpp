#include "route_map_pkg/route_map_core.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto route_map_node = std::make_shared<RouteMapNode>();
  rclcpp::spin(route_map_node);
  rclcpp::shutdown();
  return 0;
}
