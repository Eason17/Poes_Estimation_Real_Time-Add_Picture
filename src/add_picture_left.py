import cv2
import dlib
import math
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

frame = cv2.imread('face_left.jpg')
dets = detector(frame, 1)


# 无眼镜时人脸关键点标注
for k, d in enumerate(dets):
    print("dets{}".format(d))
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))

    # 使用predictor进行人脸关键点识别 shape为返回的结果
    shape = predictor(frame, d)
    # 获取第一个和第二个点的坐标（相对于图片而不是框出来的人脸）
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

    # 绘制特征点
    for index, pt in enumerate(shape.parts()):
        print('Part {}: {}'.format(index, pt))
        pt_pos = (pt.x, pt.y)
        cv2.circle(frame, pt_pos, 1, (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(index + 1), pt_pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        if index == 0:
            p1_x = pt.x
            p1_y = pt.y
        if index == 16:
            p2_x = pt.x
            p2_y = pt.y
        if index == 2:
            p3_y = pt.y
            p3_x = pt.x
        if index == 17:
            p4_y = pt.y
            p4_x = pt.x

    # 眼镜文件
    file_path = "glass_left1.png"
    logo = cv2.imread(file_path)

    logo_shape = logo.shape
    logo_height, logo_width = logo_shape[:2]

    # lower = np.array([110, 110, 115])
    lower = np.array([165, 185, 150])
    upper = np.array([170, 190, 155])
    kernel = np.ones((3, 3), np.uint8)
    cnt = 0

    # 向人脸添加眼镜
    # width = marks[16, 0] - marks[0, 0]  # 取出0和16两点的横坐标
    # height = marks[17, 1] - marks[2, 1]  # 取出3和18两点的横坐标
    width = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
    height = p4_y - p3_y
    # print(marks)
    size = (abs(int(width)) + 350, abs(int(height)) + 20)
    # size = (int(logo_width * 0.6), int(logo_height * 0.4))  # 通过检测人脸，将人脸的宽度来resize
    shrink_logo = cv2.resize(logo, size, interpolation=cv2.INTER_AREA)

    mini_frame = frame[int(p4_y) - 30:int(p4_y) - 30 + shrink_logo.shape[0],
                 int(p1_x) - 180:int(p1_x) - 180 + shrink_logo.shape[1], :]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    dil = cv2.dilate(mask, kernel, iterations=5)
    mini_dil = np.zeros_like(mini_frame)
    mini_dil[:, :, 0] = dil[int(p4_y) - 30:int(p4_y) - 30 + shrink_logo.shape[0],
                        int(p1_x) - 180:int(p1_x) - 180 + shrink_logo.shape[1]]
    mini_dil[:, :, 1] = dil[int(p4_y) - 30:int(p4_y) - 30 + shrink_logo.shape[0],
                        int(p1_x) - 180:int(p1_x) - 180 + shrink_logo.shape[1]]
    mini_dil[:, :, 2] = dil[int(p4_y) - 30:int(p4_y) - 30 + shrink_logo.shape[0],
                        int(p1_x) - 180:int(p1_x) - 180 + shrink_logo.shape[1]]

    shrink_logo_copy = shrink_logo.copy()
    shrink_logo_copy[mini_dil == 255] = 1  # 所加载的图像不会被遮盖。
    shrink_logo_copy[shrink_logo > 200] = 1  # 将白色背景利用图像掩码技术去除。
    mini_frame[shrink_logo_copy != 1] = 1
    mini_frame = mini_frame * shrink_logo_copy  # 矩阵相乘，将图像进行加合。
    frame[int(p4_y) - 30:int(p4_y) - 30 + shrink_logo.shape[0],
    int(p1_x) - 180:int(p1_x) - 180 + shrink_logo.shape[1], :] = mini_frame
    # print(marks[0, 0], '\n', frame.shape, '\n', shrink_logo.shape)
    # print(int(marks[0, 0]) + shrink_logo.shape[1], int(marks[16, 0]))
    # frame = cv2.resize(frame, (1000, 800), interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)
    cv2.imshow('mini_frame', mini_frame)
    # cv2.imwrite('./frame/' + str(cnt) + '.png', frame)
    cnt += 1


k = cv2.waitKey()
cv2.destroyAllWindows()
