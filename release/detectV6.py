from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
from math import atan2, degrees, ceil, radians, cos, sin

vehicle_model = YOLO("yolov10n.pt")  # 车辆模型
accident_model = YOLO("best.pt")  # 事故模型

video_path = "accident.mp4"  # 打开视频文件
output_path = "output.mp4"  # 输出视频路径

target_fps = 30  # 视频处理速度（帧）

max_acc = 10  # 极限加速度规定

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
    [360, 640],  # 右下角
    [360, 0]  # 左下角
])

show_matrix = False  # 打印详细指标日志
show_collision = False  # 打印碰撞框详细情况

cap = cv2.VideoCapture(video_path)

# 获取视频帧的尺寸和FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"视频宽度：{frame_width}")
print(f"视频高度：{frame_height}")

# 创建视频写入对象
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))

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


def bb_overlap(bbox1, bbox2, angle1, angle2):
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center1 = view_transformer.transform_points(np.array([center1]))[0]
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    center2 = view_transformer.transform_points(np.array([center2]))[0]

    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    max_distance = np.sqrt(a ** 2 + b ** 2)
    if distance > max_distance:
        return 0

    rect1 = create_rectangle(center1, a, b, angle1)
    rect2 = create_rectangle(center2, a, b, angle2)

    intersection_area = rect1.intersection(rect2).area

    rect_area = a * b
    overlap_ratio_1 = intersection_area / rect_area
    overlap_ratio_2 = intersection_area / rect_area

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

    if len(tracked_cars) >= 2:
        overlaps = {}
        for i in range(len(tracked_cars)):
            track_id1 = tracked_cars[i]
            if track_id1 not in boxes:
                continue
            bbox1 = boxes[track_id1]
            angle1 = calculate_angle(track_history[track_id1])
            for j in range(i + 1, len(tracked_cars)):
                track_id2 = tracked_cars[j]
                if track_id2 not in boxes:
                    continue
                bbox2 = boxes[track_id2]
                angle2 = calculate_angle(track_history[track_id2])
                overlap = bb_overlap(bbox1, bbox2, angle1, angle2)
                overlaps[(track_id1, track_id2)] = overlap

    for track_id in tracked_cars:
        if track_id not in boxes:
            continue

        track = track_history[track_id]
        acceleration = calculate_acceleration(track)
        angle = calculate_angle(track)
        max_overlap = max((bb_overlap(boxes[track_id], boxes[other_id], angle, calculate_angle(track_history[other_id]))
                           for other_id in tracked_cars if other_id != track_id and other_id in boxes), default=0)

        risk_score = calculate_risk_score(acceleration, angle, max_overlap, accident_confidence)
        risk_scores[track_id].append(risk_score)
        if show_matrix:
            print(f"ID: {track_id}")
            print(f"Acceleration: {acceleration}")
            print(f"Angle Change: {angle}")
            print(f"Overlap: {max_overlap}")
            print(f"Accident Confidence: {accident_confidence}")
            print(f"Risk_score: {risk_score}\n")
        if risk_score > 7:
            acc_detected = True
    if show_matrix:
        print("----------------------------------")

    return acc_detected


frame_interval = ceil(fps / target_fps)
frame_count = 0
frames = []

# 视频帧处理
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

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
                if x1 < acc_x2 and x2 > acc_x1 and y1 < acc_y2 and y2 > acc_y1:
                    involved_vehicles.append(track_id)
                    if show_collision:
                        print(f"Vehicle ID: {track_id}, Accident Box Confidence: {acc_confidence:.2f}")

            if involved_vehicles:
                acc_detected = detect_accidents(track_history, {id: boxes[id] for id in involved_vehicles},
                                                acc_confidence)
                if acc_detected:
                    for track_id in involved_vehicles:
                        x1, y1, x2, y2 = boxes[track_id]
                        overlay = annotated_frame.copy()
                        alpha = 0.3  # 半透明度
                        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 128, 255), -1)
                        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
                        cv2.putText(annotated_frame, "Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
    if show_collision:
        print("----------------------------")

    # 计算和显示FPS
    if vehicle_results:
        inference_time = vehicle_results[0].speed['inference'] / 1000  # 转换为秒
        fps_display = 1 / inference_time
        font_scale = frame_height / 600
        text = f'FPS: {fps_display:.2f}'
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        text_width, text_height = text_size
        text_x = frame_width - text_width - 20
        text_y = text_height + 20

        # 在帧上绘制文本
        cv2.putText(annotated_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255),
                    thickness=3)

    # 将帧写入输出视频文件
    frames.append(annotated_frame)

    # 显示带注释的帧
    resized_frame = cv2.resize(annotated_frame, (960, 540))
    cv2.imshow('Detect', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

output_frame_count = int(total_frames * (target_fps / fps))

if len(frames) < output_frame_count:
    frame_step = len(frames) / output_frame_count
    interpolated_frames = []
    for i in range(output_frame_count):
        idx = int(i * frame_step)
        interpolated_frames.append(frames[min(idx, len(frames) - 1)])
else:
    frame_step = len(frames) / output_frame_count
    for i in range(output_frame_count):
        idx = int(i * frame_step)
        out.write(frames[min(idx, len(frames) - 1)])

for frame in interpolated_frames:
    out.write(frame)

# 释放视频资源
cap.release()
out.release()
cv2.destroyAllWindows()
