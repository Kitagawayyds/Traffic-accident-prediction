from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
from math import atan2, degrees, sqrt

# 加载YOLO模型
vehicle_model = YOLO("yolov10n.pt")
accident_model = YOLO("best.pt")  # 加载事故识别模型

# 打开视频文件
video_path = "accident.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频帧的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 创建视频写入对象
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 存储轨迹历史和风险评分
track_history = defaultdict(lambda: [])
risk_scores = defaultdict(lambda: [])
accident_confidences = []  # 用于存储每一帧的事故置信度

# 计算加速度和角度变化的函数
def calculate_acceleration(track):
    accelerations = []
    if len(track) > 2:
        for i in range(2, len(track)):
            x1, y1 = track[i - 2]
            x2, y2 = track[i - 1]
            x3, y3 = track[i]
            v1 = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            v2 = sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            acceleration = v2 - v1
            accelerations.append(acceleration)
    return accelerations

def calculate_angle_change(track):
    angles = []
    if len(track) > 2:
        for i in range(2, len(track)):
            x1, y1 = track[i - 2]
            x2, y2 = track[i - 1]
            x3, y3 = track[i]
            angle1 = atan2(y2 - y1, x2 - x1)
            angle2 = atan2(y3 - y2, x3 - x2)
            angle_change = abs(degrees(angle2 - angle1))
            angles.append(angle_change)
    return angles

# 计算风险评分的函数
def calculate_risk_score(acceleration_changes, angle_changes, overlap, accident_confidence):
    def acceleration_risk(acceleration_change):
        if acceleration_change < 0.5:
            return 0
        elif acceleration_change < 1.5:
            return 3
        elif acceleration_change < 2.5:
            return 5
        else:
            return 10

    def angle_risk(angle_change):
        if angle_change < 15:
            return 0
        elif angle_change < 45:
            return 3
        elif angle_change < 75:
            return 5
        elif angle_change < 105:
            return 7
        else:
            return 10

    def overlap_risk(overlap):
        if overlap < 0.2:
            return 0
        elif overlap < 0.4:
            return 3
        elif overlap < 0.6:
            return 5
        elif overlap < 0.8:
            return 7
        else:
            return 10

    latest_acceleration_change = acceleration_changes[-1] if acceleration_changes else 0
    latest_angle_change = angle_changes[-1] if angle_changes else 0

    acceleration_score = acceleration_risk(latest_acceleration_change)
    angle_score = angle_risk(latest_angle_change)
    overlap_score = overlap_risk(overlap)

    # 加权计算总风险评分，accident_confidence是事故模型的置信度
    total_score = 0.3 * acceleration_score + 0.3 * angle_score + 0.4 * overlap_score
    weighted_risk_score = total_score * accident_confidence

    return weighted_risk_score

# 计算边界框重叠的函数
def bb_overlap(bbox1, bbox2, overlap_threshold=0.3):
    poly1 = Polygon([(bbox1[0], bbox1[1]), (bbox1[0], bbox1[3]), (bbox1[2], bbox1[3]), (bbox1[2], bbox1[1])])
    poly2 = Polygon([(bbox2[0], bbox2[1]), (bbox2[0], bbox2[3]), (bbox2[2], bbox2[3]), (bbox2[2], bbox2[1])])
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    overlap_ratio = intersection_area / union_area
    return overlap_ratio

# 事故检测函数
def detect_accidents(track_history, boxes, accident_confidence):
    collision_detected = False
    tracked_cars = list(track_history.keys())
    # print(boxes)

    # 计算当前帧中存在的车辆之间的重叠区域
    if len(tracked_cars) >= 2:
        overlaps = {}
        for i in range(len(tracked_cars)):
            track_id1 = tracked_cars[i]
            if track_id1 not in boxes:
                continue  # 如果 track_id1 不在 boxes 中，跳过
            bbox1 = boxes[track_id1]
            for j in range(i + 1, len(tracked_cars)):
                track_id2 = tracked_cars[j]
                if track_id2 not in boxes:
                    continue  # 如果 track_id2 不在 boxes 中，跳过
                bbox2 = boxes[track_id2]
                overlap = bb_overlap(bbox1, bbox2)
                overlaps[(track_id1, track_id2)] = overlap

    for track_id in tracked_cars:
        if track_id not in boxes:
            continue  # 跳过没有对应边界框的跟踪 ID

        track = track_history[track_id]
        accelerations = calculate_acceleration(track)
        angles = calculate_angle_change(track)

        # 计算当前帧中存在的车辆之间的最大重叠区域
        max_overlap = 0
        if len(tracked_cars) > 1:
            for other_id in tracked_cars:
                if other_id != track_id and other_id in boxes:
                    bbox1 = boxes[track_id]
                    bbox2 = boxes[other_id]
                    max_overlap = max(max_overlap, bb_overlap(bbox1, bbox2))

        print(f"Track ID: {track_id}")
        # print(f"Accelerations: {accelerations}")
        # print(f"Angles: {angles}")
        # print(f"Overlap: {max_overlap}")
        # print(f"Accident Confidence: {accident_confidence}")

        # 计算风险评分
        risk_score = calculate_risk_score(accelerations, angles, max_overlap, accident_confidence)
        print(f"Risk_score: {risk_score}")
        risk_scores[track_id].append(risk_score)

        if risk_score > 5:  # 超过阈值
            collision_detected = True

    return collision_detected


# 读取视频帧的循环
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # 车辆检测
        vehicle_results = vehicle_model.track(frame, persist=True, tracker='bytetrack.yaml')
        boxes = {}
        if vehicle_results[0].boxes.id is not None:
            boxes = {int(id): box.tolist() for id, box in zip(vehicle_results[0].boxes.id.cpu(), vehicle_results[0].boxes.xyxy.cpu())}
            track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()
            annotated_frame = vehicle_results[0].plot()

            for track_id in track_ids:
                x1, y1, x2, y2 = boxes[track_id]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                track = track_history[track_id]
                track.append((float(cx), float(cy)))
                if len(track) > 30:
                    track.pop(0)

                points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 255), thickness=3)

        # 事故检测
        accident_results = accident_model(frame)
        max_confidence = 0
        for result in accident_results:
            boxes_acc = result.boxes
            for box in boxes_acc:
                confidence = box.conf[0]  # 置信度
                max_confidence = max(max_confidence, confidence.item())

        accident_confidences.append(max_confidence)  # 保存每帧的事故置信度
        accident_confidence_str = f"{max_confidence:.4f}"

        # 检测事故
        collision = detect_accidents(track_history, boxes, max_confidence)
        if collision:
            cv2.putText(annotated_frame, f"Accident detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 写入视频帧
        out.write(annotated_frame)
        cv2.imshow('Frame', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
