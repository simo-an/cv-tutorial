from random import randint
from imageio import imread, imsave
from torch.autograd import Variable
from gaussian_smoothing import GaussianSmoothingNet
from sobel_filter import SobelFilterNet
from nonmax_supression import NonMaxSupression
from utils import init_parameter

import torch
import torch.nn as nn
import numpy as np

to_bw = lambda image: (image > 0.0).astype(float)

class HysteresisThresholding(nn.Module):
    def __init__(self, low_threshold=1.0, high_threshold=3.0) -> None:
        super(HysteresisThresholding, self).__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def thresholding(self, low_thresh: torch.Tensor, high_thresh: torch.Tensor):
        died = torch.zeros_like(low_thresh).squeeze()
        low_thresh = low_thresh.squeeze()
        final_image = high_thresh.squeeze().clone()

        height = final_image.shape[0] - 1 
        width = final_image.shape[1] - 1

        def connected(x, y, gap = 1):
            right = x + gap
            bottom = y + gap
            left = x - gap
            top = y - gap

            if left < 0 or top < 0 or right >= width or bottom >= height:
                return False
            
            return final_image[top, left] > 0  or final_image[top, x] > 0 or final_image[top, right] > 0 \
                or final_image[y, left] > 0 or final_image[y, right] > 0 \
                or final_image[bottom, left] > 0 or final_image[bottom, x] > 0 or final_image[bottom, right] > 0

        # 先高再宽
        def trace(x:int, y:int):
            right = x + 1
            bottom = y + 1
            left = x - 1
            top = y - 1
            if left < 0 or top < 0 or right >= width or bottom >= height or died[y, x] or final_image[y, x] > 0:
                return

            pass_high = final_image[y, x] > 0.0
            pass_low = low_thresh[y, x] > 0.0

            died[y, x] = True

            if pass_high:
                died[y, x] = False
            elif pass_low and not pass_high:
                if connected(x, y) or connected(x, y, 2): # 如果其他方向有连接
                    final_image[y, x] = low_thresh[y, x]
                    died[y, x] = False
            
            # 往回
            if final_image[y, x] > 0.0: # 当前点有连接
                if low_thresh[top, left] > 0: trace(left, top)
                if low_thresh[top, x] > 0: trace(x, top)    
                if low_thresh[top, right] > 0: trace(right, top)
                if low_thresh[y, left] > 0: trace(left, y)
                if low_thresh[bottom, left] > 0: trace(left, bottom)

            # 往下
            trace(right, y)
            trace(x, bottom)
            trace(right, bottom)
        
        for i in range(width):
            for j in range(height):
                trace(i, j)

        final_image = final_image.unsqueeze(dim=0).unsqueeze(dim=0)

        return final_image

    def forward(self, thin_edges, grad_magnitude, grad_orientation):
        low_thresholded: torch.Tensor = thin_edges.clone()
        low_thresholded[thin_edges<self.low_threshold] = 0.0

        high_threshold: torch.Tensor = thin_edges.clone()
        high_threshold[thin_edges<self.high_threshold] = 0.0

        final_thresholded = self.thresholding(low_thresholded, high_threshold)

        return low_thresholded, high_threshold, final_thresholded

if __name__ == '__main__':
    # 读取图片
    # raw_img = imread('03-exercises/coin.png') / 255.0
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
    imsave('thin_edge_img.png', thin_edge_img.data.numpy()[0,0])

    ht_net = HysteresisThresholding()
    ht_net.eval()

    low_thresholded, high_thresholded, final_thresholded = ht_net(thin_edge_img, grad_magnitude, grad_orientation)
    imsave('low_thresholded.png', to_bw(low_thresholded.data.cpu().numpy()[0, 0]))
    imsave('high_thresholded.png', to_bw(high_thresholded.data.cpu().numpy()[0, 0]))
    imsave('final_thresholded.png', to_bw(final_thresholded.data.cpu().numpy()[0, 0]))

