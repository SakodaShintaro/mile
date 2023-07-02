#include <route_map_pkg/route_map_core.hpp>

RouteMapNode::RouteMapNode() : Node("route_map_node")
{
  sub_route_ = this->create_subscription<autoware_planning_msgs::msg::LaneletRoute>(
    "/planning/mission_planning/route", 10,
    std::bind(&RouteMapNode::callback_route, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "Ready");
}

void RouteMapNode::callback_route(const autoware_planning_msgs::msg::LaneletRoute::SharedPtr msg) {
  RCLCPP_INFO(get_logger(), "callback_route");
}
