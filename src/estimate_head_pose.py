from multiprocessing import Process, Queue

import cv2
import numpy as np

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
from objloader_simple import OBJ

print("OpenCV version: {}".format(cv2.__version__))
obj = OBJ('oculos.obj', swapyz=True)

CNN_INPUT_SIZE = 128

class args():
    def __init__(self):
        self.cam = 0
        self.video = None


# 引入mark_detector来检测标志。
"""
先前报错无法序列化文件，通过debug调试找到无法序列化的模块所在的位置。
将mark_dector单独实例化，在函数get_face中设置全局变量。
"""
mark_detector = MarkDetector()


def get_face(img_queue, box_queue):
    """
    从图像队列中得到人脸，这个函数用于多处理。
    """
    global mark_detector
    while True:
        image = img_queue.get()
        box = mark_detector.extract_cnn_facebox(image)
        box_queue.put(box)


def main(obj):
    """MAIN"""
    file_path = "oculos.png"
    logo = cv2.imread(file_path)
    logo = logo[100:360, 120:550, :]  # 裁减掉眼镜图片中的空白部分

    logo_shape = logo.shape
    logo_height, logo_width = logo_shape[:2]

    # lower = np.array([110, 110, 115])
    lower = np.array([165, 185, 150])
    upper = np.array([170, 190, 155])
    kernel = np.ones((3, 3), np.uint8)
    cnt = 0

    # 视频来源从网络摄像头或视频文件。
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0
    cap = cv2.VideoCapture(video_src)  # video_src为0，表示笔记本摄像头；否则应该为传入路径。
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置分辨率
    _, sample_frame = cap.read()  # 返回布尔值和帧

    # 为多处理设置进程和队列。
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)  # 写入队列
    box_process = Process(target=get_face, args=(img_queue, box_queue,))
    # 启动一个进程，运行get_face函数，传入参数mark_detector, img_queue, box_queue。
    box_process.start()

    # 引入位姿估计器求解位姿，获得一帧来根据图像大小设置估计器。
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # 为姿态引入标量稳定器。
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()  # 设置计时器。

    while True:
        # 读取帧，剪裁，翻转，以适合你的需要。
        frame_got, frame = cap.read()  # frame_got为布尔值,frame为每一帧.
        if frame_got is False:
            break

        if video_src == 0:
            frame = cv2.flip(frame, 2)

        img_queue.put(frame)
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

            # 将标记位置从本地CNN转换为全局图像。
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # 取消后面一行的注释以显示原始标记，标记68个关键点。
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # 取消下面一行的注释以显示facebox,facebox用于标注人脸位置。
            # mark_detector.draw_box(frame, [facebox])

            # 尝试68点的姿态估计。
            pose = pose_estimator.solve_pose_by_68_points(marks)  # 在solve_pose_by_68_points方法中使用到cv2.solvePnP
            # pose中得到的是旋转矩阵和平移矩阵

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()  # 将pose转化为数组，flatten返回一个折叠成一维的数组。
            for value, ps_stb in zip(pose_np, pose_stabilizers):  # pose_stabilizers是Stabilizer类的实例。
                ps_stb.update([value])  # update更新字典中的键值对或集合中的值。
                steady_pose.append(ps_stb.state[0])  # ps_std.state是一个嵌套列表。
            steady_pose = np.reshape(steady_pose, (-1, 3))
            # print(steady_pose)

            # 取消对下一行的注释以在帧上绘制姿态注释。
            # pose_estimator.draw_annotation_box(
            #     frame, pose[0], pose[1], color=(255, 128, 128))

            # 取消对下一行的注释，以在帧上绘制稳定姿态注释。
            # pose_estimator.draw_annotation_box(
            #     frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

            # 取消对下一行的注释，以在帧上绘制坐标轴。
            # pose_estimator.draw_axes(frame, steady_pose[0], steady_pose[1])

            # add_glasses
            vertices = obj.vertices  # 读出obj文件顶点
            vertices = np.array(vertices)
            ones = np.ones(len(vertices))
            homo_vertices = np.column_stack((vertices, ones)).T  # 眼镜坐标矩阵，转换为齐次矩阵,每一列为一个点坐标
            rotation_vector = pose[0]
            translation_vector = pose[1]
            rodRotMat = cv2.Rodrigues(rotation_vector, None)  # 将旋转向量转化为旋转矩阵
            R = np.zeros((4, 4), np.float32)
            R[3, 3] = 1.0
            R[:3, :3] = rodRotMat[0]
            R[:3, 3:] = translation_vector  # 得到变换矩阵R,矩阵大小为4×4
            point_3d = np.dot(R, homo_vertices)  # 变换后的3D坐标矩阵,大小为4×n
            # print(point_3d)
            print("旋转向量：\n", rotation_vector, "\n", "平移向量：\n", translation_vector, "\n", "变换矩阵：\n", R)
            print("obj文件坐标矩阵：\n", vertices, "\n", "变换后的坐标矩阵:\n", homo_vertices)

            # add_picture
            width = marks[16, 0] - marks[0, 0]  # 取出0和16两点的横坐标
            height = marks[17, 1] - marks[2, 1]  # 取出3和18两点的横坐标
            # print(marks)
            size = (abs(int(width)) + 25, abs(int(height)))
            # size = (int(logo_width * 0.6), int(logo_height * 0.4))  # 通过检测人脸，将人脸的宽度来resize
            shrink_logo = cv2.resize(logo, size, interpolation=cv2.INTER_AREA)

            mini_frame = frame[int(marks[17, 1]):int(marks[17, 1]) + shrink_logo.shape[0],
                         int(marks[0, 0]) - 10:int(marks[0, 0]) - 10 + shrink_logo.shape[1], :]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            dil = cv2.dilate(mask, kernel, iterations=5)
            mini_dil = np.zeros_like(mini_frame)
            mini_dil[:, :, 0] = dil[int(marks[17, 1]):int(marks[17, 1]) + shrink_logo.shape[0],
                                int(marks[0, 0]) - 10:int(marks[0, 0]) - 10 + shrink_logo.shape[1]]
            mini_dil[:, :, 1] = dil[int(marks[17, 1]):int(marks[17, 1]) + shrink_logo.shape[0],
                                int(marks[0, 0]) - 10:int(marks[0, 0]) - 10 + shrink_logo.shape[1]]
            mini_dil[:, :, 2] = dil[int(marks[17, 1]):int(marks[17, 1]) + shrink_logo.shape[0],
                                int(marks[0, 0]) - 10:int(marks[0, 0]) - 10 + shrink_logo.shape[1]]

            shrink_logo_copy = shrink_logo.copy()
            shrink_logo_copy[mini_dil == 255] = 1  # 所加载的图像不会被遮盖。
            shrink_logo_copy[shrink_logo > 240] = 1  # 将白色背景利用图像掩码技术去除。
            mini_frame[shrink_logo_copy != 1] = 1
            mini_frame = mini_frame * shrink_logo_copy  # 矩阵相乘，将图像进行加合。
            frame[int(marks[17, 1]):int(marks[17, 1]) + shrink_logo.shape[0],
            int(marks[0, 0]) - 10:int(marks[0, 0]) - 10 + shrink_logo.shape[1], :] = mini_frame
            # print(marks[0, 0], '\n', frame.shape, '\n', shrink_logo.shape)
            # print(int(marks[0, 0]) + shrink_logo.shape[1], int(marks[16, 0]))
            # frame = cv2.resize(frame, (1000, 800), interpolation=cv2.INTER_AREA)
            cv2.imshow('frame', frame)
            cv2.imshow('mini_frame', mini_frame)
            # cv2.imwrite('./frame/' + str(cnt) + '.png', frame)
            cnt += 1

            if cv2.waitKey(10) == 27:
                break

            # Show preview.
            ret, new_frame = cap.read()
            new_frame = cv2.flip(new_frame, 2)

            # 取消后面一行的注释以显示原始标记，标记68个关键点。
            mark_detector.draw_marks(new_frame, marks, color=(0, 255, 0))

            # 取消对下一行的注释，以在帧上绘制稳定姿态注释。
            pose_estimator.draw_annotation_box(
                new_frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

            # 取消对下一行的注释，以在帧上绘制坐标轴。
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

    box_process.terminate()
    box_process.join()


if __name__ == '__main__':
    args = args()
    main(obj)
