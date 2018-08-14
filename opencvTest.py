import cv2
import numpy as np
import screeninfo
import datetime
import math
import time


# img = cv2.imread("E:\\Intern\\cvImgPy\\pics\\printscreen.png")
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
bg_3 = cv2.imread("E:\\Intern\\cvImgPy\\pics\\bg.png")
bg_g = cv2.cvtColor(bg_3, cv2.COLOR_BGR2GRAY)

# cap = cv2.VideoCapture(4)  # 从摄像头读取
cap = cv2.VideoCapture('E:\\Intern\\cvImgPy\\pics\\testV.MOV')  # 选择摄像头，把视频读取删除
while cap.isOpened():
    starttime = datetime.datetime.now()  # 开始计算
    ret, frame = cap.read()
    if isinstance(frame, np.ndarray):  # 因为从某一帧开始，读取的视频就不是ndarray格式的了，导致报错，所以加一个判断
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转灰度图

        hand = bg_g - gray
        _, mask = cv2.threshold(hand, 47, 255, cv2.THRESH_BINARY)

        cv2.imshow(window_name, mask)

        interval = datetime.datetime.now() - starttime
        if interval.seconds < 0.04:
            time.sleep(0.04 - interval.seconds)
        endtime = datetime.datetime.now()
        print(endtime - starttime)

    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break

cap.release()
cv2.destroyAllWindows()
