import cv2
import numpy as np
import screeninfo
import datetime
import math
import time

global frame, mask, canvas, bk, bk255, center, theta, radian, leftMg, upMg, bkW, bkH, hori_pos, sin, cos, makeupFrame
bk = (1, 1, 1)
bk255 = (255, 255, 255)
center = 0  # center的初始值，随便设置一个就好，不设置会出问题
theta = 180  # 旋转角，顺时针方向为正
radian = 2*math.pi*theta/360  # 把旋转角转换成弧度制
leftMg = 372  # 左边距
upMg = 400  # 上边距
bkW = 1176  # 白色底板宽度
hori_pos = 0.5  # 默认手的位置是底边中点
bkH = 416  # 白色底板长度
sin = math.sin(radian)
cos = math.cos(radian)


def makeMask(src, thresh):
    global mask
    bg_g = 255 * np.ones((src.shape[0], src.shape[1]), dtype=np.float32)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 转灰度图
    hand = bg_g - gray
    _, mask = cv2.threshold(hand, thresh, 255, cv2.THRESH_BINARY)  # 二值化mask, 真实猴子第一个参数要调成47，另一个视频150
    # print(mask[mask.shape[0]-1])
    return 0


def copyTo():
    global frame, canvas, mask
    locs = np.where(mask != 0)  # Get the non-zero mask locations
    y_offset = upMg + (bkH - frame.shape[0])
    x_offset = leftMg + int((bkW - frame.shape[1])/2)
    canvas[y_offset + locs[0], x_offset + locs[1]] = frame[locs[0], locs[1]]/255
    return 0


def handRotate(angle):
    global frame, mask
    frame = rotate(frame, angle, (255, 255, 255))
    mask = rotate(mask, angle, (0, 0, 0))
    return 0


# 旋转且不裁剪，但是会缩小,不用
def rotate_uncut(image, angle):
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
    rotated = cv2.warpAffine(image, M, (nW, nH))  # 纯色填充
    return rotated


# 旋转且裁剪，不会缩放
def rotate(image, angle, bV, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将底边中点设为旋转中心
    if center is None:
        center = (w / 2, h)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=bV)
    # rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated


# 找到对称的点
def point_sym(rot_center, target):  # 绕某一点旋转180度的坐标值
    x = int(2 * rot_center[0] - target[0]) + 3  # 旋转后的x坐标，后面加了修正值
    y = int(2 * rot_center[1] - target[1]) - 2  # 旋转后的y坐标，后面加了修正值
    result = (x, y)
    return result


def get_symFrame():
    global frame, makeupFrame
    makeupFrame = 255 * np.ones((2 * frame.shape[0], 3 * frame.shape[1], 3), dtype=np.float32)
    symFrame = rotate_uncut(frame, 180)
    makeupFrame[(frame.shape[0]):(2*frame.shape[0]-1), (2*center):(2*center + frame.shape[1] - 1)] = symFrame[1:, 1:] / 255  # 前h后w,两个-1除黑边
    makeupFrame[0:(frame.shape[0]), frame.shape[1]:(2 * frame.shape[1])] = frame / 255
    # cv2.imshow(window_name, makeupFrame)
    return 0

screen_id = 1
is_color = False
# get the size of the screen
screen = screeninfo.get_monitors()[screen_id]
width, height = screen.width, screen.height
window_name = 'MonkeyView'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# cap = cv2.VideoCapture(4)  # 从摄像头读取
cap = cv2.VideoCapture('E:\\Intern\\cvImgPy\\pics\\testV.MOV')  # 选择摄像头，把视频读取删除
while cap.isOpened():
    # starttime = datetime.datetime.now()  # 开始计算
    ret, frame = cap.read()
    # bg_3 = cv2.imread("E:\\Intern\\cvImgPy\\pics\\bg.png")
    if isinstance(frame, np.ndarray):  # 因为从某一帧开始，读取的视频就不是ndarray格式的了，导致报错，所以加一个判断
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        cv2.rectangle(canvas, (leftMg, upMg), (leftMg + bkW, upMg + bkH), bk, -1)  # 刷新背景
        # 清晰度变换
        # frame = cv2.medianBlur(frame, 1)  # 中值去噪,起到调节清晰度的作用

        # 尺寸变换
        # frame = cv2.resize(frame, (int(1176*1.5), int(416*1.5)), interpolation=cv2.INTER_LINEAR)
        makeMask(frame, 40)
        center = int(np.median(np.where(mask[mask.shape[0] - 1] != 0)))  # 旋转中心
        get_symFrame()
        makeMask(makeupFrame, 40)
        cv2.imshow(window_name, mask)

        # 木头or手

        # 旋转变换
        # handRotate(theta)
        # print(frame.shape[0], frame.shape[1])
        # canvas[bkH:bkH + frame.shape[0], bkW:bkW + frame.shape[1]] = frame / 255
        # copyTo()

        # cv2.imshow(window_name, mask)

        # interval = datetime.datetime.now() - starttime
        # if interval.seconds < 0.04:
        #     time.sleep(0.04 - interval.seconds)
        # time.sleep(0.1)
        # endtime = datetime.datetime.now()
        # print(endtime - starttime)

    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break

cap.release()
cv2.destroyAllWindows()
