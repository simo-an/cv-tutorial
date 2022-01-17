from imageio import imread, imsave
from torch.autograd import Variable
from gaussian_smoothing import GaussianSmoothingNet
from utils import init_parameter

import torch
import torch.nn as nn
import numpy as np

PAI = 3.1415926

class SobelFilterNet(nn.Module):
    def __init__(self) -> None:
        super(SobelFilterNet, self).__init__()
        sobel_filter = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])
        self.SFH = nn.Conv2d(1, 1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.SFV = nn.Conv2d(1, 1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)

        init_parameter(self.SFH, sobel_filter, np.array([0.0]))
        init_parameter(self.SFV, sobel_filter.T, np.array([0.0]))

    def forward(self, img):
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        # # SFH(V): sobel filter of horizontal(vertical) 水平（竖直）方向的Sobel滤波
        grad_r_x = self.SFH(img_r)  # 通道 R 的 x 方向梯度
        grad_r_y = self.SFV(img_r)
        grad_g_x = self.SFH(img_g)
        grad_g_y = self.SFV(img_g)
        grad_b_x = self.SFH(img_b)
        grad_b_y = self.SFV(img_b)

        # 计算强度（magnitude） 和 方向（orientation）
        magnitude_r = torch.sqrt(grad_r_x**2 + grad_r_y**2) # Gr^2 = Grx^2 + Gry^2
        magnitude_g = torch.sqrt(grad_g_x**2 + grad_g_y**2) 
        magnitude_b = torch.sqrt(grad_b_x**2 + grad_b_y**2)

        grad_magnitude = magnitude_r + magnitude_g + magnitude_b

        grad_y = grad_r_y + grad_g_y + grad_b_y
        grad_x = grad_r_x + grad_g_x + grad_b_x

        # tanθ = grad_y / grad_x 转化为角度 （方向角）
        grad_orientation = (torch.atan2(grad_y, grad_x) * (180.0 / PAI)) 
        grad_orientation =  torch.round(grad_orientation / 45.0) * 45.0  # 转化为 45 的倍数
        
        return grad_magnitude, grad_orientation



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

    # 保存梯度强度图
    imsave('grad_magnitude.png',grad_magnitude.data.numpy()[0,0])
