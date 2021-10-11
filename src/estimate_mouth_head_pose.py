"""

目前，人脸是由OpenCV DNN模块的检测器检测的。
然后对人脸框进行了一些修改，以适应标志检测的需要。
人脸标志的检测是使用经过TensorFlow训练的自定义卷积神经网络完成的。
然后通过求解一个PnP问题来估计头部姿态。

"""

import cv2
import numpy as np
from stabilizer import Stabilizer
from judge_pose import OperateDetect
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from multiprocessing import Process, Queue

CNN_INPUT_SIZE = 128

# 引入mark_detector来检测标志。
"""
先前报错无法序列化文件，通过debug调试找到无法序列化的模块所在的位置。
将mark_dector单独实例化，在函数get_face中设置全局变量。
"""
mark_detector = MarkDetector()
operate_detect = OperateDetect()


class args():  # 定义args类替代参数解析器，使得不用命令行也可输入。
    def __init__(self):
        self.cam = 0
        # self.cam = 1
        # self.video = "face_video.mp4"
        self.video = None


def get_face(img_queue, box_queue):
    """
    Get face from image queue. This function is used for multiprocessing.
    从图像队列中得到人脸，这个函数用于多处理。
    """
    global mark_detector
    while True:
        image = img_queue.get()
        box = mark_detector.extract_cnn_facebox(image)
        box_queue.put(box)


def main():
    # 视频来源从网络摄像头或视频文件。
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)  # video_src为0，表示笔记本摄像头；否则应该为传入路径。
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置分辨率
    _, sample_frame = cap.read()  # 返回布尔值和帧

    # Setup process and queues for multiprocessing.
    # 为多处理设置进程和队列。
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)  # 写入队列
    box_process = Process(target=get_face, args=(img_queue, box_queue,))
    # 启动一个进程，运行get_face函数，传入参数mark_detector, img_queue, box_queue.
    box_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the estimator according to the image size.
    # 引入位姿估计器求解位姿，获得一帧来根据图像大小设置估计器。
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    # 为姿态引入标量稳定器。
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()  # 设置计时器。

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()  # frame_got为布尔值,frame为每一帧.
        if frame_got is False:
            break

        # 如果帧比预期的大，就进行裁剪。
        # frame = frame[0:480, 300:940]

        # 对帧进行翻转。
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # Feed frame to image queue.
        img_queue.put(frame)

        # Get face from box queue.
        facebox = box_queue.get()

        if facebox is not None:
            # 从128x128的图像中检测标志。
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            tm.start()
            marks = mark_detector.detect_marks([face_img])  # 得到脸部标志
            tm.stop()

            # 判断眨眼、张嘴、点头、摇头
            operate_detect.detect(frame, marks)

            # 将标记位置从本地CNN转换为全局图像。
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # 尝试68点的姿态估计。
            pose = pose_estimator.solve_pose_by_68_points(marks)  # 在solve_pose_by_68_points方法中使用到cv2.solvePnP
            # pose中得到的是旋转矩阵和平移矩阵

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()  # 将pose转化为数组，flatten返回一个折叠成一维的数组。
            for value, ps_stb in zip(pose_np, pose_stabilizers):  # pose_stabilizers是Stabilizer类的实例。
                ps_stb.update([value])   # update更新字典中的键值对或集合中的值。
                steady_pose.append(ps_stb.state[0])  # ps_std.state是一个嵌套列表。
            steady_pose = np.reshape(steady_pose, (-1, 3))
            # print(steady_pose)

            # Show preview.
            ret, new_frame = cap.read()
            new_frame = cv2.flip(new_frame, 2)

            # 显示原始标记，标记68个关键点。
            mark_detector.draw_marks(new_frame, marks, color=(0, 255, 0))  # 在原先输入的视频帧中标注

            # 在帧上绘制稳定姿态注释。
            pose_estimator.draw_annotation_box(
                new_frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

            # 在帧上绘制坐标轴。
            pose_estimator.draw_axes(new_frame, steady_pose[0], steady_pose[1])

            cv2.imshow("Preview", new_frame)
            if cv2.waitKey(10) == 27:
                break

            # 显示背景为黑色的人脸标志。
            black_frame = np.zeros((400, 500, 3), np.uint8)

            # 取消后面一行的注释以显示原始标记，标记68个关键点。
            mark_detector.draw_marks(black_frame, marks, color=(0, 255, 0))

            cv2.imshow("Black", black_frame)
            if cv2.waitKey(10) == 27:
                break

    # 清除多处理过程。
    box_process.terminate()
    box_process.join()


if __name__ == '__main__':
    args = args()
    main()
