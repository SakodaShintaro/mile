import rclpy
from rclpy.node import Node


class MileNode(Node):
    def __init__(self):
        super().__init__("mile_node")
        self.timer = self.create_timer(0.1, self.on_tick)

    def on_tick(self):
        self.get_logger().info(f"Mile node")


def main(args=None):
    rclpy.init(args=args)
    node = MileNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
