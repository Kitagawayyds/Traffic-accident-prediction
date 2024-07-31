from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
from math import atan2, degrees

# 加载YOLO模型
vehicle_model = YOLO("yolov10n.pt")
accident_model = YOLO("best.pt")

# 打开视频文件
video_path = "accident.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频帧的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 创建视频写入对象
output_path = "output.mp4"
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 存储轨迹历史,风险评分,置信度等等
track_history = defaultdict(list)
risk_scores = defaultdict(list)
accident_confidences = []


# 计算加速度的函数
def calculate_acceleration(tracks):
    accelerations = []
    if len(tracks) > 2:
        for i in range(2, len(tracks)):
            (x1, y1) = tracks[i - 2]
            (x2, y2) = tracks[i - 1]
            (x3, y3) = tracks[i]

            v1 = np.linalg.norm([x2 - x1, y2 - y1])
            v2 = np.linalg.norm([x3 - x2, y3 - y2])

            acceleration = v2 - v1
            accelerations.append(acceleration)

        max_acceleration = max(abs(acc) for acc in accelerations)
        if max_acceleration != 0:
            accelerations = [acc / max_acceleration for acc in accelerations]

    return accelerations


# 计算角度的函数
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
            angle_change = min(angle_change, 360 - angle_change)
            angles.append(angle_change)
    return angles


# 计算边界框重叠的函数
def bb_overlap(bbox1, bbox2):
    poly1 = Polygon([(bbox1[0], bbox1[1]), (bbox1[0], bbox1[3]), (bbox1[2], bbox1[3]), (bbox1[2], bbox1[1])])
    poly2 = Polygon([(bbox2[0], bbox2[1]), (bbox2[0], bbox2[3]), (bbox2[2], bbox2[3]), (bbox2[2], bbox2[1])])

    intersection_area = poly1.intersection(poly2).area

    overlap_ratio_1 = intersection_area / poly1.area
    overlap_ratio_2 = intersection_area / poly2.area

    return max(overlap_ratio_1, overlap_ratio_2)


# 计算风险评分的函数
def calculate_risk_score(acceleration, angle_changes, overlap, accident_confidence):
    def acceleration_risk(acceleration):
        max_acceleration = 1
        norm_acceleration = min(acceleration / max_acceleration, 1)
        return 10 * (norm_acceleration ** 2)

    def angle_risk(angle):
        normalized_angle = angle / 180.0
        return 10 * (normalized_angle ** 3)

    def overlap_risk(overlap):
        return 10 * (np.log1p(overlap * 2) / np.log1p(2))

    latest_acceleration = acceleration[-1] if acceleration else 0
    latest_angle = angle_changes[-1] if angle_changes else 0

    acceleration_score = acceleration_risk(latest_acceleration)
    angle_score = angle_risk(latest_angle)
    overlap_score = overlap_risk(overlap)

    max_score = max(acceleration_score, angle_score, overlap_score)
    average_score = (acceleration_score + angle_score + overlap_score) / 3
    basic_risk_score = 0.5 * average_score + 0.5 * max_score
    adjusted_risk_score = basic_risk_score * (1 + (accident_confidence - 0.5) * 2)

    return np.clip(adjusted_risk_score, 0, 10)


# 事故检测函数
def detect_accidents(track_history, boxes, accident_confidence):
    collision_detected = False
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
        accelerations = calculate_acceleration(track)
        angles = calculate_angle_change(track)

        max_overlap = max((bb_overlap(boxes[track_id], boxes[other_id])
                           for other_id in tracked_cars if other_id != track_id and other_id in boxes), default=0)

        # print(f"ID: {track_id}")
        # print(f"Accelerations: {accelerations}")
        # print(f"Angles: {angles}")
        # print(f"Overlap: {max_overlap}")
        # print(f"Accident Confidence: {accident_confidence}")

        risk_score = calculate_risk_score(accelerations, angles, max_overlap, accident_confidence)
        risk_scores[track_id].append(risk_score)

        # print(f"Risk_score: {risk_score}")

        if risk_score > 7:
            collision_detected = True

    return collision_detected


# 读取视频帧的循环
while cap.isOpened():
    success, frame = cap.read()

    if success:
        vehicle_results = vehicle_model.track(frame, persist=True, tracker='botsort.yaml', verbose=False)
        boxes = {}
        if vehicle_results[0].boxes.id is not None:
            boxes = {int(id): box.tolist() for id, box in
                     zip(vehicle_results[0].boxes.id.cpu(), vehicle_results[0].boxes.xyxy.cpu())}
            track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()
            annotated_frame = vehicle_results[0].plot()

            # 车辆框
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

        accident_results = accident_model(frame, verbose=False)
        max_confidence = 0
        accident_boxes = []

        # 事故框
        for result in accident_results:
            boxes_acc = result.boxes
            for box in boxes_acc:
                confidence = box.conf[0]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                max_confidence = max(max_confidence, confidence.item())
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                confidence_text = f"{confidence.item():.2f}"
                cv2.putText(annotated_frame, confidence_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
                accident_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        accident_confidences.append(max_confidence)
        collision_detected = detect_accidents(track_history, boxes, max_confidence)

        # 红框显示
        if collision_detected:
            for (x1, y1, x2, y2) in accident_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, "Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # FPS计算和显示
        if vehicle_results:
            inference_time = vehicle_results[0].speed['inference'] / 1000  # 转换为秒
            fps_display = 1 / inference_time

        cv2.putText(annotated_frame, f"FPS: {fps_display:.2f}", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        out.write(annotated_frame)
        cv2.imshow("Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
