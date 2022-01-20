from random import randint
from tkinter.messagebox import NO
from turtle import pen, right
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

        height = final_image.shape[0]
        width = final_image.shape[1]
        '''
        def trace_right_bottom():
            for x in range(1, width-1):
                right = x + 1
                for y in range(1, height-1):
                    bottom = y + 1
                    if final_image[y, x]: # 当前点有连接
                        if low_thresh[y, right] > 0.0: final_image[y, right] = low_thresh[y, right]        # 正右
                        if low_thresh[bottom, x] > 0.0: final_image[bottom, x] = low_thresh[bottom, x]       # 正下
                        if low_thresh[bottom, right] > 0.0: final_image[bottom, right] = low_thresh[bottom, right]   # 右下
        
        def trace_left_top():
            for x in range(1, width-1):
                cx = width - x
                left = cx - 1
                for y in range(1, height-1):
                    cy = height - y
                    top = cy - 1
                    if final_image[cy, cx]: # 当前点有连接
                        if low_thresh[cy, left] > 0.0: final_image[cy, left] = low_thresh[cy, left]        # 正左
                        if low_thresh[top, cx] > 0.0: final_image[top, cx] = low_thresh[top, cx]       # 正上
                        if low_thresh[top, left] > 0.0: final_image[top, left] = low_thresh[top, left]   # 左上

        def trace_right_top():
            for x in range(1, width-1):
                cx = width - x
                left = cx - 1
                for y in range(1, height-1):
                    bottom = y + 1
                    if final_image[y, cx]: # 当前点有连接
                        if low_thresh[y, left] > 0.0: final_image[y, left] = low_thresh[y, left]        # 正左
                        if low_thresh[bottom, cx] > 0.0: final_image[bottom, cx] = low_thresh[bottom, cx]       # 正下
                        if low_thresh[bottom, left] > 0.0: final_image[bottom, left] = low_thresh[bottom, left]   # 左下

        def trace_left_bottom():
            for x in range(1, width-1):
                right = x + 1
                for y in range(1, height-1):
                    cy = height - y
                    top = cy - 1
                    if final_image[cy, x]: # 当前点有连接
                        if low_thresh[cy, right] > 0.0: final_image[cy, right] = low_thresh[cy, right]        # 正右
                        if low_thresh[top, x] > 0.0: final_image[top, x] = low_thresh[top, x]       # 正上
                        if low_thresh[top, right] > 0.0: final_image[top, right] = low_thresh[top, right]   # 右上

        '''
        def trace(direction):
            if not direction: return
            for x in range(1, width - 1):
                cx = x
                if direction == 'left-top' or direction == 'left-bottom':
                    cx = width - 1 - x
                left = cx - 1
                right = cx + 1
                for y in range(1, height - 1):
                    cy = y
                    if direction == 'left-top' or direction == 'right-top':
                        cy = height - 1 - y
                    top = cy - 1
                    bottom = cy + 1
                    if final_image[cy, cx]: # 当前点有连接
                        if low_thresh[top, left] > 0.0: final_image[top, left] = low_thresh[top, left]   # 左上
                        if low_thresh[top, cx] > 0.0: final_image[top, cx] = low_thresh[top, cx]       # 正上
                        if low_thresh[top, right] > 0.0: final_image[top, right] = low_thresh[top, right]   # 右上
                        
                        if low_thresh[cy, left] > 0.0: final_image[cy, left] = low_thresh[cy, left]        # 正左
                        if low_thresh[cy, right] > 0.0: final_image[cy, right] = low_thresh[cy, right]        # 正右
                        
                        if low_thresh[bottom, left] > 0.0: final_image[bottom, left] = low_thresh[bottom, left]   # 左下
                        if low_thresh[bottom, cx] > 0.0: final_image[bottom, cx] = low_thresh[bottom, cx]       # 正下
                        if low_thresh[bottom, right] > 0.0: final_image[bottom, right] = low_thresh[bottom, right]   # 右下


        trace('right-bottom')
        trace('left-top')
        trace('right-top')
        trace('left-bottom')


        final_image = final_image.unsqueeze(dim=0).unsqueeze(dim=0)

        return final_image

    def forward(self, thin_edges):
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
    low_thresholded, high_thresholded, final_thresholded = ht_net(thin_edge_img)

    imsave('low_thresholded.png', to_bw(low_thresholded.data.cpu().numpy()[0, 0]))
    imsave('high_thresholded.png', to_bw(high_thresholded.data.cpu().numpy()[0, 0]))
    imsave('final_thresholded.png', to_bw(final_thresholded.data.cpu().numpy()[0, 0]))

