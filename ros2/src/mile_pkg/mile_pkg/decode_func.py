import numpy as np
import torch


def tensor_to_image(ts_image: torch.Tensor) -> np.ndarray:
    """
    ts_image: torch.Size([1, 1, 1, 192, 192])
    """
    ts_image = ts_image.squeeze(0)
    ts_image = ts_image.squeeze(0)
    ts_image = ts_image.permute([1, 2, 0])
    ts_image = ts_image.to(torch.uint8)
    image = ts_image.cpu().numpy()
    return image


def decode_segmap(x: torch.Tensor, nc=8) -> np.ndarray:
    """
    x: torch.Size([1, 1, 8, 192, 192])
    """
    label_colors = np.array([(0, 0, 0),
                             (128, 0, 0),
                             (0, 128, 0),
                             (128, 128, 0),
                             (0, 0, 128),
                             (128, 0, 128),
                             (0, 128, 128),
                             (128, 128, 128)])

    h, w = x.shape[3], x.shape[4]
    x = x.squeeze(0)
    x = x.squeeze(0)
    x = x.argmax(0)

    r = np.zeros((h, w)).astype(np.uint8)
    g = np.zeros((h, w)).astype(np.uint8)
    b = np.zeros((h, w)).astype(np.uint8)

    for l in range(0, nc):
        idx = (x == l)
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


if __name__ == "__main__":
    bev_instance_center_1 = torch.randn([1, 1, 1, 192, 192])
    bev_instance_offset_1 = torch.randn([1, 1, 1, 192, 192])
    bev_segmentation_1 = torch.randn([1, 1, 8, 192, 192])

    image_instance_center_1 = tensor_to_image(bev_instance_center_1)
    image_instance_offset_1 = tensor_to_image(bev_instance_offset_1)
    image_segmentation_1 = decode_segmap(bev_segmentation_1)

    print(type(image_instance_center_1), image_instance_center_1.shape)
    print(type(image_instance_offset_1), image_instance_offset_1.shape)
    print(type(image_segmentation_1), image_segmentation_1.shape)

    import cv2
    cv2.imwrite("image_instance_center_1.png", image_instance_center_1)
    cv2.imwrite("image_instance_offset_1.png", image_instance_offset_1)
    cv2.imwrite("image_segmentation_1.png", image_segmentation_1)
