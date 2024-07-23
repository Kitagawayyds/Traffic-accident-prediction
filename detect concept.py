from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
from math import atan2, degrees, sqrt

# 加载YOLO模型
model = YOLO("yolov10n.pt")

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

# 计算加速度和角度变化的函数
def calculate_acceleration(track):
    accelerations = []
    if len(track) > 2:
        for i in range(2, len(track)):
            x1, y1 = track[i-2]
            x2, y2 = track[i-1]
            x3, y3 = track[i]
            v1 = sqrt((x2 - x1)**2 + (y2 - y1)**2)
            v2 = sqrt((x3 - x2)**2 + (y3 - y2)**2)
            acceleration = v2 - v1
            accelerations.append(acceleration)
    return accelerations

def calculate_angle_change(track):
    angles = []
    if len(track) > 2:
        for i in range(2, len(track)):
            x1, y1 = track[i-2]
            x2, y2 = track[i-1]
            x3, y3 = track[i]
            angle1 = atan2(y2 - y1, x2 - x1)
            angle2 = atan2(y3 - y2, x3 - x2)
            angle_change = abs(degrees(angle2 - angle1))
            angles.append(angle_change)
    return angles

# 计算风险评分的函数
def calculate_risk_score(acceleration_changes, angle_changes, overlap):
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

    total_score = 0.4 * acceleration_score + 0.3 * angle_score + 0.3 * overlap_score
    return total_score

# 碰撞检测函数
def check_for_collision(track_history, boxes):
    collision_detected = False
    for track_id1, track1 in track_history.items():
        if track_id1 not in boxes:
            continue
        for track_id2, track2 in track_history.items():
            if track_id1 != track_id2 and track_id2 in boxes:
                bbox1 = boxes[track_id1]
                bbox2 = boxes[track_id2]
                overlap = bb_overlap(bbox1, bbox2)
                accelerations1 = calculate_acceleration(track1)
                angles1 = calculate_angle_change(track1)
                accelerations2 = calculate_acceleration(track2)
                angles2 = calculate_angle_change(track2)

                risk_score1 = calculate_risk_score(accelerations1, angles1, overlap)
                risk_score2 = calculate_risk_score(accelerations2, angles2, overlap)

                risk_scores[track_id1].append(risk_score1)
                risk_scores[track_id2].append(risk_score2)

                if risk_score1 > 5 or risk_score2 > 5:  # 超过阈值
                    collision_detected = True
                    return collision_detected, track_id1, track_id2
    return collision_detected, None, None

# 计算边界框重叠的函数
def bb_overlap(bbox1, bbox2, overlap_threshold=0.3):
    poly1 = Polygon([(bbox1[0], bbox1[1]), (bbox1[0], bbox1[3]), (bbox1[2], bbox1[3]), (bbox1[2], bbox1[1])])
    poly2 = Polygon([(bbox2[0], bbox2[1]), (bbox2[0], bbox2[3]), (bbox2[2], bbox2[3]), (bbox2[2], bbox2[1])])
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    overlap_ratio = intersection_area / union_area
    return overlap_ratio

# 读取视频帧的循环
while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, tracker='bytetrack.yaml')

        if results[0].boxes.id is not None:
            boxes = {int(id): box.tolist() for id, box in zip(results[0].boxes.id.cpu(), results[0].boxes.xyxy.cpu())}
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot()

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

            collision, id1, id2 = check_for_collision(track_history, boxes)
            if collision:
                print(f"Collision detected between ID: {id1} and ID: {id2}")
                cv2.putText(annotated_frame, f"Collision between ID: {id1} and ID: {id2}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 写入视频帧
            out.write(annotated_frame)
            cv2.imshow("Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# 将轨迹历史和风险评分保存到txt文件
with open("track_history.txt", "w") as f:
    for track_id, track in track_history.items():
        accelerations = calculate_acceleration(track)
        angles = calculate_angle_change(track)
        overlap = 0  # Initialize overlap to 0, as it's only relevant during collisions
        risk_score_series = [calculate_risk_score(accelerations[:i+1], angles[:i+1], overlap) for i in range(len(accelerations))]
        f.write(f"ID: {track_id}, Track: {track}, Accelerations: {accelerations}, Angles: {angles}, Risk Scores: {risk_score_series}\n")
