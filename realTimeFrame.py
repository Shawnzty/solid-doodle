import cv2
import numpy as np
import screeninfo
import datetime
import math

global bk, bk255, center, theta, radian, leftMg, upMg, bkW, bkH, hori_pos
bk = (163/255, 169/255, 164/255)  # 底板颜色 除以255
bk255 = (163, 169, 164)  # 底板颜色
# bk = (1, 1, 1)
# bk255 = (255, 255, 255)
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


# 主要的处理部分
def frame():
    # 标识全局变量
    global center
    # 窗口运行程序
    screen_id = 1
    is_color = False

    # get the size of the screen
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height
    canvas = np.zeros((height, width, 3), dtype=np.float32)  # 创建黑色背景
    window_name = 'MonkeyView'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # cap = cv2.VideoCapture(4)  # 从摄像头读取
    cap = cv2.VideoCapture('pics/testV2.MOV')  # 选择摄像头，把视频读取删除
    while cap.isOpened():
        ret, frame = cap.read()
        if isinstance(frame, np.ndarray):  # 因为从某一帧开始，读取的视频就不是ndarray格式的了，导致报错，所以加一个判断

            starttime = datetime.datetime.now()  # 开始计算

            cv2.rectangle(canvas, (leftMg, upMg), (1360, 880), bk, -1)  # 刷新背景

            frame = resize(frame, width=700)  # 大小缩放

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转灰度图

            # 抓取手的部分
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            image, contours, hierarchy = cv2.findContours(thresh, 3, 2)  # 提取轮廓，本来是3，2
            cnt = contours[0]
            # 外接矩形
            x, y, w, h = cv2.boundingRect(cnt)
            # binary = image[y:y + h, x:x + w]  # 裁剪二值图像
            center = bottomCenter(image[y:y + h, x:x + w], center)   # 找到底边中点

            # frame = cv2.medianBlur(frame, 1)  # 中值去噪,起到调节清晰度的作用
            # cut = frame[y:y + h, x:x + w]  # 裁剪真实图像

            # 只旋转手部区域，似乎没有成功
            for i in range(y, y + h):
                for j in range(x, x + w):
                    if image[i][j] == 255:
                        new_point = point_rot(x + j, y + i, x + center, y + h)
                        canvas[new_point[1], new_point[0]] = frame[i, j]

            # cut = rotate_bound(cut, theta)  # 旋转并不裁剪

            # 计算旋转贴图的位置
            # if radian >= 0:
            #     x_offset = int(leftMg + hori_pos * bkW - center * math.cos(radian))  # 左上角的X坐标
            #     y_offset = int(upMg + bkH - h * math.cos(radian) - center * math.sin(radian))  # 左上角的Y坐标
            # else:
            #     cal_rad = -radian  # 取正
            #     x_offset = int(leftMg + hori_pos * bkW - (h * math.sin(cal_rad) + center * math.cos(cal_rad)))
            #     y_offset = int(upMg + bkH - (h * math.cos(cal_rad) + (w - center) * math.sin(cal_rad)))
            # canvas[y_offset:y_offset+cut.shape[0], x_offset:x_offset+cut.shape[1]] = cut/255  # 把转好了的手画上去

            # 补缺失的三角形部分
            # if radian > 0:
            #     rot_center = (leftMg + hori_pos * bkW, upMg + bkH)
            #     start_point = (int(rot_center[0] - center * math.cos(radian)), int(rot_center[1]) - center * math.sin(radian))
            #     for k in range(0, int(center * math.sin(radian))+1):
            #         for j in range(int(-k * math.tan(radian)), int(k * (1 / math.tan(radian)))+3):  # 一行一行替换，范围要加修正
            #             x = int(start_point[0] + j)
            #             y = int(start_point[1] + k)
            #             get_pixel = point_sym(rot_center, (x, y))
            #             # print(canvas[get_pixel[1], get_pixel[0], 1])
            #             if canvas[get_pixel[1], get_pixel[0], 1] == 0:
            #                 canvas[y, x] = bk
            #             else:
            #                 canvas[y, x] = canvas[get_pixel[1], get_pixel[0]]
            # elif radian < 0:
            #     break

            # cv2.line(canvas, (leftMg + bkH, upMg), (leftMg + bkH, upMg + bkH), (255, 0, 0), 3)
            # cv2.circle(canvas, (leftMg + 300, upMg + 200), 20, (0, 0, 1), -1)
            # cv2.circle(canvas, (leftMg + 400, upMg + 300), 20, (1, 0, 0), -1)

            cv2.rectangle(canvas, (0, 880), (1920, 1080), (0, 0, 0), -1)  # 底部贴一个黑色的矩形
            cv2.imshow(window_name, canvas)
            endtime = datetime.datetime.now()
            print(endtime - starttime)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            break

    cap.release()
    cv2.destroyAllWindows()
