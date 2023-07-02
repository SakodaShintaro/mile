#ifndef ROUTE_MAP_PKG__ROUTE_MAP_CORE_HPP_
#define ROUTE_MAP_PKG__ROUTE_MAP_CORE_HPP_

#include <rclcpp/rclcpp.hpp>

#include <autoware_planning_msgs/msg/lanelet_route.hpp>

class RouteMapNode : public rclcpp::Node
{
public:
  RouteMapNode();

private:
  void callback_route(const autoware_planning_msgs::msg::LaneletRoute::SharedPtr msg);
  rclcpp::Subscription<autoware_planning_msgs::msg::LaneletRoute>::SharedPtr sub_route_;
};

#endif
