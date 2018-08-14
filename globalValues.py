import math

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