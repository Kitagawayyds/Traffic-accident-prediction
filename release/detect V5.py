from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
from math import atan2, degrees

vehicle_model = YOLO("yolov10n.pt")  # 车辆模型

accident_model = YOLO("best.pt")  # 事故模型

video_path = "accident.mp4"  # 打开视频文件

output_path = "output.mp4"  # 输出视频路径

max_acc = 10  # 极限加速度规定

a = 100  # 长轴(汽车)
b = 100  # 短轴(汽车)

SOURCE = np.array([  # 初始化视角转换器，定义源坐标和目标坐标
    [0, 0],  # 左上角
    [0, 640],  # 右上角
    [360, 640],  # 右下角
    [360, 0]  # 左下角
])

TARGET = np.array([
    [0, 0],  # 左上角
    [0, 640],  # 右上角
    [360, 640],  # 右下角
    [360, 0]  # 左下角
])

show = False  # 打印详细日志

cap = cv2.VideoCapture(video_path)

# 获取视频帧的尺寸和FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# print(frame_width)
# print(frame_height)

# 创建视频写入对象
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 存储跟踪历史记录、风险评分、事故置信度等
track_history = defaultdict(lambda: deque(maxlen=30))
risk_scores = defaultdict(lambda: deque(maxlen=30))
accident_confidences = deque(maxlen=30)


# 定义视角转换器类
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


view_transformer = ViewTransformer(source=SOURCE, target=TARGET)


# 计算加速度
def calculate_acceleration(track):
    if len(track) < 3:
        return 0
    recent_track = view_transformer.transform_points(np.array(track)[-3:])
    velocities = np.linalg.norm(np.diff(recent_track, axis=0), axis=1)
    latest_velocity = velocities[-1]
    previous_velocity = velocities[-2]
    acceleration = latest_velocity - previous_velocity
    return np.clip(acceleration / max_acc, 0, 1)


# 计算角度变化
def calculate_angle(track):
    if len(track) < 3:
        return 0
    recent_track = view_transformer.transform_points(np.array(track)[-3:])
    p1, p2, p3 = recent_track
    angle1 = atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle2 = atan2(p3[1] - p2[1], p3[0] - p2[0])
    angle_change = abs(degrees(angle2 - angle1))
    angle_change = min(angle_change, 360 - angle_change)
    return angle_change


# 计算重叠度
def ellipse_intersection_area(e1, e2):
    e1_polygon = e1
    e2_polygon = e2
    intersection_area = e1_polygon.intersection(e2_polygon).area
    return intersection_area


def create_ellipse(center):
    angle = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + a * np.cos(angle)
    y = center[1] + b * np.sin(angle)
    points = np.vstack((x, y)).T
    return Polygon(points)


def bb_overlap(bbox1, bbox2):
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center1 = view_transformer.transform_points(np.array([center1]))[0]
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    center2 = view_transformer.transform_points(np.array([center2]))[0]

    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    if distance > a:
        return 0

    ellipse1 = create_ellipse(center1)
    ellipse2 = create_ellipse(center2)

    intersection_area = ellipse_intersection_area(ellipse1, ellipse2)

    overlap_ratio_1 = intersection_area / (np.pi * a * b)
    overlap_ratio_2 = intersection_area / (np.pi * a * b)
    return max(overlap_ratio_1, overlap_ratio_2)


# 计算风险评分
def calculate_risk_score(acceleration, angle, overlap, accident_confidence):
    def acceleration_risk(acc):
        return 10 * (acc ** 2)

    def angle_risk(angle):
        norm_angle = angle / 180.0
        return 10 * (norm_angle ** 3)

    def overlap_risk(ovlp):
        return 10 * (np.log1p(ovlp * 2) / np.log1p(2))

    acceleration_score = acceleration_risk(acceleration)
    angle_score = angle_risk(angle)
    overlap_score = overlap_risk(overlap)

    max_score = max(acceleration_score, angle_score, overlap_score)
    average_score = (acceleration_score + angle_score + overlap_score) / 3
    basic_risk_score = 0.5 * average_score + 0.5 * max_score
    adjusted_risk_score = basic_risk_score * (1 + (accident_confidence - 0.5) * 2)

    return np.clip(adjusted_risk_score, 0, 10)


# 事故检测
def detect_accidents(track_history, boxes, accident_confidence):
    acc_detected = False
    tracked_cars = list(track_history.keys())

    # print(boxes)

    if len(tracked_cars) >= 2:
        overlaps = {}
        for i in range(len(tracked_cars)):
            track_id1 = tracked_cars[i]
            if track_id1 not in boxes:
                continue
            bbox1 = boxes[track_id1]
            for j in range(i + 1, len(tracked_cars)):
                track_id2 = tracked_cars[j]
                if track_id2 not in boxes:
                    continue
                bbox2 = boxes[track_id2]
                overlap = bb_overlap(bbox1, bbox2)
                overlaps[(track_id1, track_id2)] = overlap

    for track_id in tracked_cars:
        if track_id not in boxes:
            continue

        track = track_history[track_id]
        acceleration = calculate_acceleration(track)
        angle = calculate_angle(track)
        max_overlap = max((bb_overlap(boxes[track_id], boxes[other_id])
                           for other_id in tracked_cars if other_id != track_id and other_id in boxes), default=0)

        risk_score = calculate_risk_score(acceleration, angle, max_overlap, accident_confidence)
        risk_scores[track_id].append(risk_score)
        if show:
            print(f"ID: {track_id}")
            print(f"Acceleration: {acceleration}")
            print(f"Angle Change: {angle}")
            print(f"Overlap: {max_overlap}")
            print(f"Accident Confidence: {accident_confidence}")
            print(f"Risk_score: {risk_score}\n")
        if risk_score > 7:
            acc_detected = True
    if show:
        print("----------------------------------")

    return acc_detected


# 视频帧处理循环
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    annotated_frame = frame.copy()

    # 车辆检测
    vehicle_results = vehicle_model.track(frame, persist=True, tracker='botsort.yaml', verbose=False)
    boxes = {}
    if vehicle_results[0].boxes.id is not None:
        boxes = {int(id): box.tolist() for id, box in
                 zip(vehicle_results[0].boxes.id.cpu(), vehicle_results[0].boxes.xyxy.cpu())}
        track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()
        annotated_frame = vehicle_results[0].plot()

        # 跟踪车辆
        for track_id in track_ids:
            x1, y1, x2, y2 = boxes[track_id]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            track = track_history[track_id]
            track.append((float(cx), float(cy)))
            points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 255), thickness=3)

    # 事故检测
    accident_results = accident_model(frame, verbose=False)
    accident_boxes = []
    detected_objects = False
    acc_detected = False
    involved_vehicles = set()

    for result in accident_results:
        boxes_acc = result.boxes
        for box in boxes_acc:
            confidence = box.conf[0]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            confidence_text = f"{confidence.item():.2f}"
            cv2.putText(annotated_frame, confidence_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            accident_boxes.append((int(x1), int(y1), int(x2), int(y2), confidence.item()))
            detected_objects = True

    if detected_objects:
        for acc_box in accident_boxes:
            acc_x1, acc_y1, acc_x2, acc_y2, acc_confidence = acc_box
            involved_vehicles = []

            for track_id, box in boxes.items():
                x1, y1, x2, y2 = box
                if (x1 < acc_x2 and x2 > acc_x1 and y1 < acc_y2 and y2 > acc_y1):
                    involved_vehicles.append(track_id)
                    # print(f"Vehicle ID: {track_id}, Accident Box Confidence: {acc_confidence:.2f}")

            if involved_vehicles:
                acc_detected = detect_accidents(track_history, {id: boxes[id] for id in involved_vehicles},
                                                acc_confidence)
                if acc_detected:
                    for track_id in involved_vehicles:
                        x1, y1, x2, y2 = boxes[track_id]
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, "Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
    # print("----------------------------")

    # 计算和显示FPS
    if vehicle_results:
        inference_time = vehicle_results[0].speed['inference'] / 1000  # 转换为秒
        fps_display = 1 / inference_time
        cv2.putText(annotated_frame, f'FPS: {fps_display:.2f}', (frame_width - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

    # 将帧写入输出视频文件
    out.write(annotated_frame)

    # 显示带注释的帧
    cv2.imshow('Detect', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频资源
cap.release()
out.release()
cv2.destroyAllWindows()
