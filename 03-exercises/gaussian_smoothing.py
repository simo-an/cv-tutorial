from imageio import imread, imsave
from torch.autograd import Variable
from scipy.signal import gaussian

import torch
import torch.nn as nn
import numpy as np

def init_parameters(layer, w, b):
    layer.weight.data.copy_(torch.from_numpy(w))
    layer.bias.data.copy_(torch.from_numpy(b))

class GaussianSmoothingNet(nn.Module):
    def __init__(self) -> None:
        super(GaussianSmoothingNet, self).__init__()

        filter_size = 5
        # shape为(1, 5), 方差为 1.0 的高斯滤波核
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size]) 

        # GFH(V): gaussian filter of horizontal(vertical) 水平（竖直）方向的高斯滤波核
        self.GFH = nn.Conv2d(1, 1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.GFV = nn.Conv2d(1, 1, kernel_size=(filter_size,1), padding=(filter_size//2,0))

        # 设置 w 的值为 高斯平滑核, b 的值为 0.0
        init_parameters(self.GFH, generated_filters, np.array([0.0])) 
        init_parameters(self.GFV, generated_filters.T, np.array([0.0])) 

    def forward(self, img):
        img_r = img[:,0:1]  # 取出RGB三个通道的数据
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        # 对图片的三个通道进行水平、垂直滤波
        blurred_img_r = self.GFV(self.GFH(img_r))
        blurred_img_g = self.GFV(self.GFH(img_g))
        blurred_img_b = self.GFV(self.GFH(img_b))

        # 合并成一张图
        blurred_img = torch.stack([blurred_img_r, blurred_img_g, blurred_img_b], dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        return blurred_img

if __name__ == '__main__':
    # 读取图片
    raw_img = imread('coin.png') / 255.0
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    batch = torch.stack([img]).float()

    # 高斯去噪
    net = GaussianSmoothingNet()
    net.eval()
    denoising_img = net(Variable(batch))

    # 保存图片
    imsave('gaussian_smoothing.png',denoising_img.data.numpy()[0,0])
