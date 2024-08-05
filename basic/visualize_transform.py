import cv2
import numpy as np
from shapely.geometry import Polygon
from ultralytics import YOLO
from math import atan2, degrees, radians, cos, sin

# 读取视频并处理
video_path = 'accident.mp4'
output_path = 'transform.mp4'

a = 100  # 长(汽车)
b = 50  # 宽(汽车)

SOURCE = np.array([  # 初始化视角转换器，定义源坐标和目标坐标
    [0, 0],  # 左上角
    [0, 640],  # 右上角
    [360, 640],  # 右下角
    [360, 0]  # 左下角
])

TARGET = np.array([
    [0, 0],  # 左上角
    [0, 640],  # 右上角
    [300, 640],  # 右下角
    [300, 0]  # 左下角
])

vehicle_model = YOLO("yolov10n.pt")  # 车辆模型


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


# 初始化 ViewTransformer
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)


# 计算角度变化
def calculate_angle(track):
    if len(track) < 3:
        return 0
    recent_track = view_transformer.transform_points(np.array(track)[-3:])
    p1, p2, p3 = recent_track
    angle1 = atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle2 = atan2(p3[1] - p2[1], p3[0] - p2[0])
    angle = abs(degrees(angle2 - angle1))
    angle = min(angle, 360 - angle)
    return angle


# 计算重叠度
def create_rectangle(center, width, height, angle):
    cx, cy = center
    angle = radians(angle)
    rect_points = np.array([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2]
    ])
    rotation_matrix = np.array([
        [cos(angle), -sin(angle)],
        [sin(angle), cos(angle)]
    ])
    rotated_points = np.dot(rect_points, rotation_matrix)
    rotated_points[:, 0] += cx
    rotated_points[:, 1] += cy
    return Polygon(rotated_points)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height))

# 初始化车辆跟踪记录
track_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 创建白色背景
    background_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

    # 获取车辆跟踪结果
    vehicle_results = vehicle_model.track(frame, persist=True, tracker='botsort.yaml', verbose=False)
    boxes = {}
    if vehicle_results[0].boxes.id is not None:
        boxes = {int(id): box.tolist() for id, box in
                 zip(vehicle_results[0].boxes.id.cpu(), vehicle_results[0].boxes.xyxy.cpu())}
        track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()

        # 跟踪车辆
        for track_id in track_ids:
            x1, y1, x2, y2 = boxes[track_id]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # 更新车辆轨迹
            if track_id not in track_history:
                track_history[track_id] = []
            track = track_history[track_id]
            track.append((float(cx), float(cy)))
            points = np.array(track).astype(np.int32).reshape((-1, 1, 2))

            # 计算车辆的角度变化
            angle = calculate_angle(track)

            # 计算车辆框
            polygon = create_rectangle((cx, cy), a, b, angle)

            # 绘制车辆框（用多边形表示）
            polygon_points = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.polylines(background_frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

            # 绘制轨迹
            cv2.polylines(background_frame, [points], isClosed=False, color=(255, 255, 255), thickness=3)

    # 绘制目标区域框
    x, y, w, h = 0, 0, 640, 300
    cv2.rectangle(background_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 合并原始帧和处理后的帧
    combined_frame = np.hstack((frame, background_frame))

    # 显示合并的帧
    cv2.imshow('Original and Processed Frames', combined_frame)

    # 写入输出视频
    out.write(combined_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
