from imageio import imread, imsave
import numpy as np
import cv2 as cv

'''
img: 图片
win_size: 滑动窗口大小
k: 常数
thresh: 判断为角点的阈值
'''
def detect_corners(img:np.ndarray, win_size:int, k:float, thresh:int):
    dy, dx = np.gradient(img)

    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2

    offset = win_size//2
    height = img.shape[0]
    width = img.shape[1]

    # 记录角点坐标
    corners = []

    # 滑动窗口
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            win_Ixx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1].sum()
            win_Ixy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1].sum()
            win_Iyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1].sum()

            det = win_Ixx * win_Iyy - win_Ixy * win_Ixy
            tr = win_Ixx + win_Iyy

            r = det - k * tr

            if r > thresh:
                corners.append((x, y))
    

    return corners


def change_color(img:np.ndarray, point, RGB):
    img[point[1], point[0], 0] = RGB[0]
    img[point[1], point[0], 1] = RGB[1]
    img[point[1], point[0], 2] = RGB[2]

def main():
    # 定义一些超参
    win_size = 8
    k = 0.1
    thresh = 0

    # 读取图片
    rgb_img:np.ndarray = imread('desk.png')
    gray_img:np.ndarray = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY) / 255.0
    # gray_img:np.ndarray = imread('f.png') / 255.0

    corners = detect_corners(gray_img, win_size, k, thresh)

    for corner in corners:
        change_color(rgb_img, corner, (255, 0, 0))
    
    imsave('temp.png', rgb_img)

if __name__ == "__main__":
    main()