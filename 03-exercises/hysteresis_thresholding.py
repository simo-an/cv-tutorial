from imageio import imread, imsave
from torch.autograd import Variable
from gaussian_smoothing import GaussianSmoothingNet
from sobel_filter import SobelFilterNet
from nonmax_supression import NonMaxSupression
from utils import init_parameter

import torch
import torch.nn as nn
import numpy as np


class HysteresisThresholding(nn.Module):
    def __init__(self, threshold=3.0) -> None:
        super(HysteresisThresholding, self).__init__()
        self.threshold = threshold

    def forward(self, thin_edges, grad_magnitude, grad_orientation):
        thresholded: torch.Tensor = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0

        early_threshold = grad_magnitude.clone()
        early_threshold[grad_magnitude<self.threshold] = 0.0

        return thresholded


if __name__ == '__main__':
    # 读取图片
    raw_img = imread('coin.png') / 255.0
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    batch = torch.stack([img]).float()

    # 高斯去噪
    gaussian_net = GaussianSmoothingNet()
    gaussian_net.eval()
    denoising_img = gaussian_net(Variable(batch))

    # Sobel 计算梯度
    sobel_net = SobelFilterNet()
    sobel_net.eval()
    grad_magnitude, grad_orientation = sobel_net(denoising_img)

    # 进行非极大化抑制
    nms_net = NonMaxSupression()
    nms_net.eval()
    thin_edge_img = nms_net(grad_magnitude, grad_orientation)
    imsave('thin_edge_img.png',thin_edge_img.data.numpy()[0,0])

    ht_net = HysteresisThresholding()
    ht_net.eval()

    thresholded = ht_net(thin_edge_img, grad_magnitude, grad_orientation)
    imsave('final.png', (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float))

