from imageio import imread, imsave
from torch.autograd import Variable

import torch
import torch.nn as nn
import numpy as np

def init_parameter(layer, w, b):
    layer.weight.data.copy_(torch.from_numpy(w))
    layer.bias.data.copy_(torch.from_numpy(b))

class NonMaxSupression(nn.Module):
    def __init__(self) -> None:
        super(NonMaxSupression, self).__init__()
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

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])
        
        self.directional_filter = nn.Conv2d(1, 8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)

        init_parameter(self.directional_filter, all_filters[:, None, ...], np.zeros(shape=(all_filters.shape[0],)))


    def forward(self, grad_magnitude):
        all_filtered = self.directional_filter(grad_magnitude)
        print(all_filtered)
        print(all_filtered.view(-1))


# if __name__ == '__main__':
#     # 进行非极大化抑制
#     grad_magnitude = torch.ones((1, 1, 3, 3))
#     print(grad_magnitude)
#     nms_net = NonMaxSupression()
#     nms_net.eval()
#     nms_net(Variable(grad_magnitude))

X = np.array(range(12)).reshape((3, 4))

print(X)
print(X[:, 0:2])    