import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist


class OperateDetect:

    def __init__(self):
        self.weight = "shape_predictor_68_face_landmarks.dat"
        self.MOUTH_AR_THRESH = 0.2
        self.Nod_threshold = 0.01
        self.shake_threshold = 0.01
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.weight)
        self.nStart, self.nEnd = 31, 35  # 鼻子对应的索引
        self.compare_point = [0, 0]  # 刚开始的时候设置鼻子的中点在[0, 0]

    @staticmethod
    def mouth_aspect_ratio(marks):
        left_side_height = dist.euclidean(marks[61], marks[67])  # 嘴巴左侧竖直高度
        right_side_height = dist.euclidean(marks[63], marks[65])  # 嘴巴右侧竖直高度
        mouth_width = dist.euclidean(marks[48], marks[54])  # 嘴巴宽度
        mouth_ratio = (left_side_height + right_side_height) / (2.0 * mouth_width)  # 嘴巴纵横比
        return mouth_ratio

    @staticmethod
    def center_point(nose):
        return nose.mean(axis=0)

    @staticmethod
    def nod_aspect_ratio(size, pre_point, now_point):
        return abs(float((pre_point[1] - now_point[1]) / (size[0] / 2)))

    @staticmethod
    def shake_aspect_ratio(size, pre_point, now_point):
        return abs(float((pre_point[0] - now_point[0]) / (size[1] / 2)))

    def nod_shark(self, size, shape):
        nose = shape[self.nStart:self.nEnd]
        nose_center = self.center_point(nose)
        nod_value, shake_value = 0, 0
        if self.compare_point[0] != 0:
            nod_value = self.nod_aspect_ratio(size, nose_center, self.compare_point)
            shake_value = self.shake_aspect_ratio(size, nose_center, self.compare_point)
        self.compare_point = nose_center
        return nod_value, shake_value

    # 三个数值分别对应着张嘴、点头和摇头
    def action_judgment(self, action_value):
        action_type = np.array([0, 0, 0])
        mar, nod_value, shake_value = action_value
        if mar > self.MOUTH_AR_THRESH:
            action_type[0] = 1
        if nod_value > self.Nod_threshold:
            action_type[1] = 1
        if shake_value > self.shake_threshold:
            action_type[2] = 1
        return action_type

    def detect(self, frame, marks):
        img = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        size = frame.shape
        if len(rects) != 0:
            shape = self.predictor(img, rects[0])
            shape = face_utils.shape_to_np(shape)
            mar = self.mouth_aspect_ratio(marks)
            nod_value, shake_value = self.nod_shark(size, shape)
            act_type = self.action_judgment((mar, nod_value, shake_value))
            print(act_type)
