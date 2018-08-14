import cv2
import numpy as np
import math
import screeninfo

# 默认值
global bk, bk255, center, theta, radian, leftMg, upMg, bkW, bkH, hori_pos
# bk = (163/255, 169/255, 164/255)  # 底板颜色 除以255
# bk255 = (163, 169, 164)  # 底板颜色
bk = (1, 1, 1)
bk255 = (255, 255, 255)
center = 0  # center的初始值，随便设置一个就好，不设置会出问题
theta = 30  # 旋转角，顺时针方向为正
radian = 2*math.pi*theta/360  # 把旋转角转换成弧度制
leftMg = 560  # 左边距
upMg = 480  # 上边距
bkW = 800  # 白色底板宽度
hori_pos = 0.5  # 默认手的位置是底边中点
bkH = 400  # 白色底板长度


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 初始化缩放比例，并获取图像尺寸
    dim = None
    (h, w) = image.shape[:2]

    # 如果宽度和高度均为0，则返回原图
    if width is None and height is None:
        return image

    # 宽度是0
    if width is None:
        # 则根据高度计算缩放比例
        r = height / float(h)
        dim = (int(w * r), height)

    # 如果高度为0
    else:
        # 根据宽度计算缩放比例
        r = width / float(w)
        dim = (width, int(h * r))

    # 缩放图像
    resized = cv2.resize(image, dim, interpolation=inter)

    # 返回缩放后的图像
    return resized


def sharpen(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


def bottomCenter(image, center):  # 输入一个二值化图像，返回底边中心位置的序号
    side = []
    row = image.shape[0] - 1
    for i in range(0, image.shape[1] - 1):
        if image[row][i] == 255:
            side.append(i)
    if len(side) < 5:  # 防止越界错误
        return center
    else:
        return side[int(len(side) / 2)]


def rotate_bound(image, angle):  # 旋转不裁剪
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w/2, h/2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=bk255)  # 纯色填充
    # return cv2.warpAffine(image, M, (w, h), borderMode=1)  # 边缘像素填充


def point_sym(rot_center, target):  # 绕某一点旋转180度的坐标值
    x = int(2 * rot_center[0] - target[0]) + 3  # 旋转后的x坐标，后面加了修正值
    y = int(2 * rot_center[1] - target[1]) - 2  # 旋转后的y坐标，后面加了修正值
    result = (x, y)
    return result


def point_rot(ax, ay, ox, oy):  # 绕某一点旋转任意角度
    bx = int((ax - ox) * math.cos(radian) - (ay - oy) * math.sin(radian) + ox)
    by = int((ax - ox) * math.sin(radian) + (ay - oy) * math.cos(radian) + oy)
    result = (bx, by)
    return result
