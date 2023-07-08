import rclpy
from rclpy.node import Node
import torch
from omegaconf import OmegaConf
from mile.config import get_cfg
from mile.models.mile import Mile
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from autoware_auto_planning_msgs.msg import Trajectory


class MileNode(Node):
    def __init__(self):
        super().__init__("mile_node")
        self.declare_parameter("ckpt_path", "")
        self.declare_parameter("conf_path", "")
        ckpt_path = self.get_parameter("ckpt_path").value
        conf_path = self.get_parameter("conf_path").value
        self.get_logger().info(f"ckpt path : {ckpt_path}")
        self.get_logger().info(f"conf path : {conf_path}")
        assert ckpt_path, "ckpt_path is None"
        assert conf_path, "conf_path is None"

        cfg = OmegaConf.load(conf_path)
        cfg = OmegaConf.to_container(cfg)
        cfg = get_cfg(cfg_dict=cfg)
        self.get_logger().info(f"load config : {cfg}")

        self.model = Mile(cfg)
        checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
        checkpoint = {key[6:]: value for key,
                      value in checkpoint.items() if key[:5] == 'model'}
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()
        self.get_logger().info(f"loaded weights")

        """
        Make dummy input following format
        ----------
            batch: dict of torch.Tensor
                keys:
                    image: (b, s, 3, h, w)
                    route_map: (b, s, 3, h_r, w_r)
                    speed: (b, s, 1)
                    intrinsics: (b, s, 3, 3)
                    extrinsics: (b, s, 4, 4)
                    throttle_brake: (b, s, 1)
                    steering: (b, s, 1)
        """
        b = 1
        s = 1
        self.batch = {}
        self.batch['image'] = torch.zeros((b, s, 3, 224, 224))
        self.batch['route_map'] = torch.zeros((b, s, 3, 224, 224))
        self.batch['speed'] = torch.zeros((b, s, 1))
        self.batch['intrinsics'] = torch.zeros((b, s, 3, 3))
        self.batch['extrinsics'] = torch.zeros((b, s, 4, 4))
        self.batch['throttle_brake'] = torch.zeros((b, s, 1))
        self.batch['steering'] = torch.zeros((b, s, 1))
        self.H = 224
        self.W = 224

        """
        dict_keys(['bev_instance_center_1', 'bev_instance_center_2', 'bev_instance_center_4',
                'bev_instance_offset_1', 'bev_instance_offset_2', 'bev_instance_offset_4',
                'bev_segmentation_1', 'bev_segmentation_2', 'bev_segmentation_4',
                'posterior', 'prior', 'steering', 'throttle_brake'])
        """

        # get from ros2 topic
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(
            Image, "/sensing/camera/traffic_light/image_raw", self.image_callback, qos_profile)
        self.sub_trajectory = self.create_subscription(
            Trajectory, "/planning/scenario_planning/trajectory", self.trajectory_callback, qos_profile)

        self.ready_image = False

        self.get_logger().info(f"Ready")

    def try_infer(self):
        if not self.ready_image:
            return
        self.ready_image = False
        out = self.model(self.batch)
        self.get_logger().info(f"out.keys() = {out.keys()}")

    def image_callback(self, msg: Image):
        # self.get_logger().info(f"Subscribe Image")
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        ts_image = torch.tensor(cv_image)
        ts_image = ts_image.permute([2, 0, 1])
        ts_image = ts_image.to(torch.float32)
        ts_image = ts_image.unsqueeze(0)
        ts_image = torch.nn.functional.interpolate(
            ts_image, size=(self.H, self.W), mode="bilinear")
        ts_image = ts_image.unsqueeze(0)
        self.batch['image'] = ts_image
        # self.get_logger().info(f"Image Shape = {ts_image.shape}")
        self.ready_image = True
        self.try_infer()

    def trajectory_callback(self, msg: Trajectory):
        self.get_logger().info(f"trajectory = {len(msg.points)}")


def main(args=None):
    rclpy.init(args=args)
    node = MileNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
