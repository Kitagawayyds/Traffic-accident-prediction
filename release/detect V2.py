from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
from math import atan2, degrees

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

    # 计算交集和并集的面积
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    # 计算两个方向上的重叠率
    overlap_ratio_1 = intersection_area / poly1.area
    overlap_ratio_2 = intersection_area / poly2.area

    # 返回两个重叠率中的最大值
    overlap_ratio = max(overlap_ratio_1, overlap_ratio_2)

    return overlap_ratio


# 计算风险评分的函数
def calculate_risk_score(acceleration, angle_changes, overlap, accident_confidence):
    def acceleration_risk(acceleration):
        max_acceleration = 1
        norm_acceleration = min(acceleration / max_acceleration, 1)
        risk_score = 10 * (norm_acceleration ** 2)
        return risk_score

    def angle_risk(angle):
        normalized_angle = angle / 180.0
        risk = normalized_angle ** 3
        risk_score = risk * 10
        return risk_score

    def overlap_risk(overlap):
        risk_score = 10 * (np.log1p(overlap * 2) / np.log1p(2))  # Adjusted log function based on alpha
        return risk_score

    latest_acceleration = acceleration[-1] if acceleration else 0
    latest_angle = angle_changes[-1] if angle_changes else 0

    acceleration_score = acceleration_risk(latest_acceleration)
    angle_score = angle_risk(latest_angle)
    overlap_score = overlap_risk(overlap)

    # 加权计算总风险评分，accident_confidence是事故模型的置信度
    # 确定输入分数中的最大分数
    max_score = np.max([acceleration_score, angle_score, overlap_score])

    # 计算分数的平均值
    average_score = (acceleration_score + angle_score + overlap_score) / 3

    # 计算基本风险分数，作为分数的加权平均
    basic_risk_score = 0.5 * average_score + 0.5 * max_score

    # 调整风险分数，以体现置信度的主导地位
    # 当置信度较高时，基本风险分数影响较大；当置信度较低时，基本风险分数影响较小
    adjusted_risk_score = basic_risk_score * (1 + (accident_confidence - 0.5) * 2)

    # 计算最终加权风险分数
    # 确保最终风险分数在合理范围内
    weighted_risk_score = np.clip(adjusted_risk_score, 0, 10)

    return weighted_risk_score
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

        # print(f"ID: {track_id}")
        # print(f"Accelerations: {accelerations}")
        # print(f"Angles: {angles}")
        # print(f"Overlap: {max_overlap}")
        # print(f"Accident Confidence: {accident_confidence}")

        # 计算风险评分
        risk_score = calculate_risk_score(accelerations, angles, max_overlap, accident_confidence)
        # print(f"Risk_score: {risk_score}")
        risk_scores[track_id].append(risk_score)

        if risk_score > 7:  # 超过阈值
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
            boxes = {int(id): box.tolist() for id, box in
                     zip(vehicle_results[0].boxes.id.cpu(), vehicle_results[0].boxes.xyxy.cpu())}
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
        accident_boxes = []  # 存储事故区域的边界框

        for result in accident_results:
            boxes_acc = result.boxes
            for box in boxes_acc:
                confidence = box.conf[0]  # 置信度
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # 获取边界框的坐标
                max_confidence = max(max_confidence, confidence.item())
                # 绘制事故模型的边界框
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 绘制置信度值
                confidence_text = f"{confidence.item():.2f}"
                cv2.putText(annotated_frame, confidence_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
                # 记录事故区域的边界框
                accident_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        accident_confidences.append(max_confidence)  # 保存每帧的事故置信度
        accident_confidence_str = f"{max_confidence:.4f}"

        # 检测事故
        collision = detect_accidents(track_history, boxes, max_confidence)
        if collision:
            for (x1, y1, x2, y2) in accident_boxes:
                # 绘制红色框标记事故区域
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 绘制事故检测文本
            cv2.putText(annotated_frame, "Accident detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
