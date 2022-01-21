from imageio import imread, imsave
from torch.autograd import Variable
from gaussian_smoothing import GaussianSmoothingNet
from sobel_filter import SobelFilterNet
from utils import init_parameter

import torch
import torch.nn as nn
import numpy as np

# Roberts 算子用于检测各种角度的边
filter_0 = np.array([   [ 0, 0, 0],
                        [ 0, 1,-1],
                        [ 0, 0, 0]])

filter_45 = np.array([  [ 0, 0, 0],
                        [ 0, 1, 0],
                        [ 0, 0,-1]])

filter_90 = np.array([  [ 0, 0, 0],
                        [ 0, 1, 0],
                        [ 0,-1, 0]])

filter_135 = np.array([ [ 0, 0, 0],
                        [ 0, 1, 0],
                        [-1, 0, 0]])

filter_180 = np.array([ [ 0, 0, 0],
                        [-1, 1, 0],
                        [ 0, 0, 0]])

filter_225 = np.array([ [-1, 0, 0],
                        [ 0, 1, 0],
                        [ 0, 0, 0]])

filter_270 = np.array([ [ 0,-1, 0],
                        [ 0, 1, 0],
                        [ 0, 0, 0]])

filter_315 = np.array([ [ 0, 0,-1],
                        [ 0, 1, 0],
                        [ 0, 0, 0]])

class NonMaxSupression(nn.Module):
    def __init__(self) -> None:
        super(NonMaxSupression, self).__init__()

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])
        
        '''
        directional_filter功能见下面详细说明
        '''
        self.directional_filter = nn.Conv2d(1, 8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)

        init_parameter(self.directional_filter, all_filters[:, None, ...], np.zeros(shape=(all_filters.shape[0],)))

    def forward(self, grad_magnitude, grad_orientation):

        all_orient_magnitude = self.directional_filter(grad_magnitude)     # 当前点梯度分别与其其他8个方向邻域点做差（相当于二阶梯度）

        '''
                \ 3|2 /
                 \ | /
            4     \|/    1
        -----------|------------
            5     /|\    8
                 / | \ 
                / 6|7 \ 
        注: 各个区域都是45度
        '''

        positive_orient = (grad_orientation / 45) % 8             # 设置正方向的类型，一共有八种不同类型的方向
        negative_orient = ((grad_orientation / 45) + 4) % 8       # +4 = 4 * 45 = 180 即旋转180度(如 1 -(+4)-> 5)

        height = positive_orient.size()[2]                        # 得到图片的宽高
        width = positive_orient.size()[3]
        pixel_count = height * width                                # 计算图片所有的像素点数
        pixel_offset = torch.FloatTensor([range(pixel_count)])

        position = (positive_orient.view(-1).data * pixel_count + pixel_offset).squeeze() # 角度 * 像素数 + 像素所在位置

        # 拿到图像中所有点与其正向邻域点的梯度的梯度（当前点梯度 - 正向邻域点梯度，根据其值与0的大小判断当前点是不是邻域内最大的）
        channel_select_filtered_positive = all_orient_magnitude.view(-1)[position.long()].view(1, height, width)

        position = (negative_orient.view(-1).data * pixel_count + pixel_offset).squeeze()

        # 拿到图像中所有点与其反向邻域点的梯度的梯度
        channel_select_filtered_negative = all_orient_magnitude.view(-1)[position.long()].view(1, height, width)

        # 组合成两个通道
        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0 # 如果min{当前梯度-正向点梯度, 当前梯度-反向点梯度} > 0，则当前梯度最大
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_magnitude.clone()
        thin_edges[is_max==0] = 0.0

        return thin_edges


if __name__ == '__main__':
    # 读取图片
    raw_img = imread('03-exercises/coin.png') / 255.0
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
