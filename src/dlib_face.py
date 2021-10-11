import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

frame = cv2.imread('face.png')
dets = detector(frame, 1)
black_frame = np.zeros((350, 350, 3), np.uint8)


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
        cv2.circle(black_frame, pt_pos, 1, (0, 255, 0), 1)

cv2.imshow('img', frame)
cv2.imshow('black', black_frame)
k = cv2.waitKey()
cv2.destroyAllWindows()
