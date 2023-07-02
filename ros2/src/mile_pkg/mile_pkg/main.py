import rclpy
from rclpy.node import Node
import torch
from omegaconf import OmegaConf
from mile.config import get_cfg
from mile.models.mile import Mile


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

        self.timer = self.create_timer(0.1, self.on_tick)

        cfg = OmegaConf.load(conf_path)
        print(f"step1 cfg = {cfg}")

        cfg = OmegaConf.to_container(cfg)
        print(f"step2 cfg = {cfg}")

        cfg = get_cfg(cfg_dict=cfg)
        print(f"step3 cfg = {cfg}")

        model = Mile(cfg)

        checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
        checkpoint = {key[6:]: value for key, value in checkpoint.items() if key[:5] == 'model'}

        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        print(model)
        print(f'Loaded weights')

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
        batch = {}
        batch['image'] = torch.zeros((b, s, 3, 224, 224))
        batch['route_map'] = torch.zeros((b, s, 3, 224, 224))
        batch['speed'] = torch.zeros((b, s, 1))
        batch['intrinsics'] = torch.zeros((b, s, 3, 3))
        batch['extrinsics'] = torch.zeros((b, s, 4, 4))
        batch['throttle_brake'] = torch.zeros((b, s, 1))
        batch['steering'] = torch.zeros((b, s, 1))

        out = model(batch)
        print(sorted(list(out.keys())))
        """
        dict_keys(['bev_instance_center_1', 'bev_instance_center_2', 'bev_instance_center_4',
                'bev_instance_offset_1', 'bev_instance_offset_2', 'bev_instance_offset_4',
                'bev_segmentation_1', 'bev_segmentation_2', 'bev_segmentation_4',
                'posterior', 'prior', 'steering', 'throttle_brake'])
        """


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
