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
from tf2_ros import TransformListener
from tf2_ros.buffer import Buffer
import tf2_geometry_msgs
import numpy as np
from mile_pkg.decode_func import tensor_to_image, decode_segmap


class MileNode(Node):
    def __init__(self):
        super().__init__("mile_node")
        self.get_logger().info(f"Initializing ...")
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
        self.pub_route_map_image = self.create_publisher(
            Image, "/mile/route_map_image", 10)
        self.pub_bev_instance_center_1 = self.create_publisher(
            Image, "/mile/bev_instance_center_1", 10)
        self.pub_bev_instance_offset_1 = self.create_publisher(
            Image, "/mile/bev_instance_offset_1", 10)
        self.pub_bev_segmentation_1 = self.create_publisher(
            Image, "/mile/bev_segmentation_1", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.ready_image = False
        self.ready_route_map = False

        self.get_logger().info(f"Ready")

    @torch.no_grad()
    def try_infer(self):
        if not self.ready_image or not self.ready_route_map:
            return
        self.ready_image = False
        self.ready_route_map = False
        out = self.model(self.batch)
        """
        dict_keys(['bev_instance_center_1', 'bev_instance_center_2', 'bev_instance_center_4',
                'bev_instance_offset_1', 'bev_instance_offset_2', 'bev_instance_offset_4',
                'bev_segmentation_1', 'bev_segmentation_2', 'bev_segmentation_4',
                'posterior', 'prior', 'steering', 'throttle_brake'])
        bev_instance_center_1 = torch.Size([1, 1, 1, 192, 192])
        bev_instance_center_2 = torch.Size([1, 1, 1, 96, 96])
        bev_instance_center_4 = torch.Size([1, 1, 1, 48, 48])
        bev_instance_offset_1 = torch.Size([1, 1, 1, 192, 192])
        bev_instance_offset_2 = torch.Size([1, 1, 1, 96, 96])
        bev_instance_offset_4 = torch.Size([1, 1, 1, 48, 48])
        bev_segmentation_1 = torch.Size([1, 1, 8, 192, 192])
        bev_segmentation_2 = torch.Size([1, 1, 8, 96, 96])
        bev_segmentation_4 = torch.Size([1, 1, 8, 48, 48])
        """
        self.get_logger().info(f"out.keys() = {out.keys()}")
        bev_instance_center_1 = out["bev_instance_center_1"]
        bev_instance_offset_1 = out["bev_instance_offset_1"]
        bev_segmentation_1 = out["bev_segmentation_1"]
        self.get_logger().info(
            f"bev_instance_center_1 = {bev_instance_center_1.shape}")
        self.get_logger().info(
            f"bev_instance_offset_1 = {bev_instance_offset_1.shape}")
        self.get_logger().info(
            f"bev_segmentation_1 = {bev_segmentation_1.shape}")
        bev_instance_center_1 = tensor_to_image(bev_instance_center_1)
        bev_instance_offset_1 = tensor_to_image(bev_instance_offset_1)
        bev_segmentation_1 = decode_segmap(bev_segmentation_1)
        self.get_logger().info(
            f"bev_instance_center_1 = {bev_instance_center_1.shape}")
        self.get_logger().info(
            f"bev_instance_offset_1 = {bev_instance_offset_1.shape}")
        self.get_logger().info(
            f"bev_segmentation_1 = {bev_segmentation_1.shape}")
        msg = self.bridge.cv2_to_imgmsg(bev_instance_center_1, "mono8")
        self.pub_bev_instance_center_1.publish(msg)
        # msg = self.bridge.cv2_to_imgmsg(bev_instance_offset_1, "mono8")
        # self.pub_bev_instance_offset_1.publish(msg)
        msg = self.bridge.cv2_to_imgmsg(bev_segmentation_1, "rgb8")
        self.pub_bev_segmentation_1.publish(msg)
        self.get_logger().info(f"posterior = {out['posterior'].keys()}")
        self.get_logger().info(f"prior = {out['prior'].keys()}")
        self.get_logger().info(f"steering = {out['steering'].shape}")
        self.get_logger().info(
            f"throttle_brake = {out['throttle_brake'].shape}")

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
        # get transform map frame to base_link frame
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link", "map", msg.header.stamp)
            self.get_logger().info(f"transform = {transform}")
        except Exception as e:
            self.get_logger().info(f"Exception lookup_transform: {e}")
            return
        route_map_image = np.zeros((self.H, self.W), np.uint8)
        for point in msg.points:
            point_pose = point.pose
            try:
                transformed_pose = tf2_geometry_msgs.do_transform_pose(
                    point_pose, transform)
            except Exception as e:
                self.get_logger().info(f"Exception transform: {e}")
                return
            WIDTH = 50
            pixel_x = int((-transformed_pose.position.y +
                          WIDTH / 2) * self.W // (WIDTH))
            pixel_y = int((-transformed_pose.position.x + WIDTH)
                          * self.H // WIDTH)
            if 0 <= pixel_x < self.W and 0 <= pixel_y < self.H:
                route_map_image[pixel_y, pixel_x] = 255

        # convert rgb
        route_map_image = np.stack(
            [route_map_image, route_map_image, route_map_image], axis=2)

        # set to batch
        self.batch['route_map'] = torch.tensor(route_map_image).permute(
            [2, 0, 1]).unsqueeze(0).unsqueeze(0).to(torch.float32)

        # publish
        msg = self.bridge.cv2_to_imgmsg(route_map_image, "rgb8")
        self.pub_route_map_image.publish(msg)
        self.ready_route_map = True
        self.try_infer()


def main(args=None):
    rclpy.init(args=args)
    node = MileNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
