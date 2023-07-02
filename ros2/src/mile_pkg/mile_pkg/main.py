import rclpy
from rclpy.node import Node


class MileNode(Node):
    def __init__(self):
        super().__init__("mile_node")
        self.declare_parameter("ckpt_path")
        self.ckpt_path = self.get_parameter("ckpt_path").value
        self.get_logger().info(f"Ckpt path : {self.ckpt_path}")
        assert self.ckpt_path is not None, "self.ckpt_path is None"

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
