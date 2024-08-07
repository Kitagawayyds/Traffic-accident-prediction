import time
from collections import defaultdict, deque
import cv2
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from shapely.geometry import Polygon
from math import atan2, degrees, ceil, radians, cos, sin

vehicle_model = YOLO("yolov10n.pt")  # 车辆模型
accident_model = YOLO("best.pt")  # 事故模型

video_path = "accident2.mp4"  # 打开视频文件
output_path = "output.mp4"  # 输出视频路径
transform_path = 'transform.mp4'  # 映射视频路径

target_fps = 10  # 视频处理速度（帧）

max_acc = 10  # 极限加速度规定

vehicle_length = 4  # 长(汽车)
vehicle_width = 2  # 宽(汽车)

sigma = 2  # 轨迹平滑系数

# 映射区域
SOURCE = np.array([
    [554, 308],
    [954, 310],
    [1364, 826],
    [412, 826]
])

TARGET = np.array([
    [0, 0],
    [6, 0],
    [6, 23],
    [0, 23]
])

scale_factor = 30  # 目标区域放大比例

margin = 50  # 留白距离

TARGET = TARGET * scale_factor
vehicle_width = vehicle_width * scale_factor
vehicle_length = vehicle_length * scale_factor

'''
假设这是视频：
+-------x-------+
|(0,0)     (x,0)|    
y               |
|(0,y)     (x,y)|
+---------------+
'''

show_matrix = False  # 打印车辆详细日志
show_collision = False  # 打印碰撞框详细情况
show_track = False  # 绘制车辆轨迹
show_transform = True  # 启用映射可视化
save = False  # 保存推理视频
smooth = True  # 平滑轨迹

cap = cv2.VideoCapture(video_path)

# 获取视频帧的尺寸和FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width_t = np.abs(TARGET[1, 0] - TARGET[0, 0])
height_t = np.abs(TARGET[2, 1] - TARGET[1, 1])

# 参数配置
config_data = [
    ["Parameter", "Value"],
    ["Video Path", video_path],
    ["Output Video Path", output_path],
    ["Transform Video Path", transform_path],
    ["Vehicle Model", vehicle_model.model_name],
    ["Accident Model", accident_model.model_name],
    ["Source Region", SOURCE],
    ["Target Region", TARGET],
    ["Scale Factor", scale_factor],
    ["Smooth Factor", sigma],
    ["Margin", margin],
    ["Target FPS", target_fps],
    ["Max Acceleration", max_acc],
    ["Vehicle Length", vehicle_length],
    ["Vehicle Width", vehicle_width],
    ["Show Detailed Metrics", bool(show_matrix)],
    ["Show Collision Box Details", bool(show_collision)],
    ["Show Vehicle Track", bool(show_track)],
    ["Transformation Visualization", bool(show_transform)],
    ["Save Output", bool(save)],
    ["Smooth track", bool(sigma)],
    ["Video Width", frame_width],
    ["Video Height", frame_height],
    [f"Mapping Width", width_t],
    [f"Mapping Height", height_t],
    ["Video FPS", fps],
    ["Total Frames", total_frames]
]

table = PrettyTable()
table.field_names = config_data[0]
for row in config_data[1:]:
    table.add_row(row)
software_info = """
🚥 ARMS V7, made by Linhan Song
If you want to get the latest version and the detailed information of the program, please visit:
https://github.com/Kitagawayyds/Traffic-accident-prediction
"""
print(software_info)
print(table)

# 创建视频写入对象
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
if save:
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))
if show_transform:
    out_t = cv2.VideoWriter(transform_path, fourcc, target_fps, (width_t + margin * 2, height_t + margin * 2))

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


# 平滑轨迹
def smooth_trajectory(track):
    if len(track) < 3:
        return track

    x_coords = [p[0] for p in track]
    y_coords = [p[1] for p in track]

    x_smooth = gaussian_filter1d(x_coords, sigma)
    y_smooth = gaussian_filter1d(y_coords, sigma)

    smoothed_track = list(zip(x_smooth, y_smooth))

    return smoothed_track


# 计算加速度
def calculate_acceleration(track):
    if len(track) < 3:
        return 0
    recent_track = view_transformer.transform_points(np.array(track)[-3:])
    velocities = np.linalg.norm(np.diff(recent_track, axis=0), axis=1) * target_fps / scale_factor * 3.6
    latest_velocity = velocities[-1]
    previous_velocity = velocities[-2]
    acceleration = latest_velocity - previous_velocity
    return np.clip(acceleration / max_acc, 0, 1)


# 计算角度变化
def calculate_angle(track):
    if len(track) < 3:
        return 0, 0
    recent_track = view_transformer.transform_points(np.array(track)[-3:])
    p1, p2, p3 = recent_track
    angle1 = atan2(p1[0] - p2[0], p1[1] - p2[1])
    angle2 = atan2(p2[0] - p3[0], p2[1] - p3[1])
    angle_change = abs(degrees(angle2 - angle1))
    angle_change = min(angle_change, 360 - angle_change)
    current_angle = degrees(angle2)
    return angle_change, current_angle


# 碰撞框生成
def create_rectangle(center, length, width, angle):
    cx, cy = center
    angle = radians(angle)
    rect_points = np.array([
        [-width / 2, -length / 2],
        [width / 2, -length / 2],
        [width / 2, length / 2],
        [-width / 2, length / 2]
    ])
    rotation_matrix = np.array([
        [cos(angle), -sin(angle)],
        [sin(angle), cos(angle)]
    ])
    rotated_points = np.dot(rect_points, rotation_matrix)
    rotated_points[:, 0] += cx
    rotated_points[:, 1] += cy
    return Polygon(rotated_points)


# 计算重叠度
def bb_overlap(bbox1, bbox2, angle1, angle2):
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center1 = view_transformer.transform_points(np.array([center1]))[0]
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    center2 = view_transformer.transform_points(np.array([center2]))[0]

    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    max_distance = np.sqrt(vehicle_length ** 2 + vehicle_width ** 2)
    if distance > max_distance:
        return 0

    rect1 = create_rectangle(center1, vehicle_length, vehicle_width, angle1)
    rect2 = create_rectangle(center2, vehicle_length, vehicle_width, angle2)

    intersection_area = rect1.intersection(rect2).area

    rect_area = vehicle_length * vehicle_width
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

    if show_matrix:
        table_m = PrettyTable()
        table_m.field_names = ["ID", "Acceleration", "Angle Change", "Max Overlap", "Accident Confidence", "Risk Score"]

    if len(tracked_cars) >= 2:
        overlaps = {}
        for i in range(len(tracked_cars)):
            track_id1 = tracked_cars[i]
            if track_id1 not in boxes:
                continue
            bbox1 = boxes[track_id1]
            angle1 = calculate_angle(track_history[track_id1])[1]
            for j in range(i + 1, len(tracked_cars)):
                track_id2 = tracked_cars[j]
                if track_id2 not in boxes:
                    continue
                bbox2 = boxes[track_id2]
                angle2 = calculate_angle(track_history[track_id2])[1]
                overlap = bb_overlap(bbox1, bbox2, angle1, angle2)
                overlaps[(track_id1, track_id2)] = overlap

    for track_id in tracked_cars:
        if track_id not in boxes:
            continue

        track = track_history[track_id]
        acceleration = calculate_acceleration(track)
        angle_change, current_angle = calculate_angle(track)
        max_overlap = max(
            (bb_overlap(boxes[track_id], boxes[other_id], current_angle, calculate_angle(track_history[other_id])[1])
             for other_id in tracked_cars if other_id != track_id and other_id in boxes), default=0)

        risk_score = calculate_risk_score(acceleration, angle_change, max_overlap, accident_confidence)
        risk_scores[track_id].append(risk_score)
        if show_matrix:
            table_m.add_row([
                track_id,
                f"{acceleration: .4f}",
                f"{angle_change: .4f}",
                f"{max_overlap: .4f}",
                f"{accident_confidence: .4f}",
                f"{risk_score: .4f}"
            ])
        if risk_score > 7:
            acc_detected = True

    if show_matrix:
        print(table_m)
        print("\n")

    return acc_detected


start_program_time = time.time()
frame_interval = ceil(fps / target_fps)
frame_count = 0
frames = []

# 视频帧处理
for _ in tqdm(range(total_frames), desc="Processing"):
    success, frame = cap.read()
    if not success:
        break

    if show_transform:
        frame_t = np.ones((height_t + margin * 2, width_t + margin * 2, 3), dtype=np.uint8) * 255

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
            if smooth:
                track = smooth_trajectory(track)
            if show_track:
                points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 255), thickness=3)

            if show_transform:
                angle = calculate_angle(track)[1]
                center = (cx, cy)
                center = view_transformer.transform_points(np.array([center]))[0]
                center[0] = center[0] + margin
                center[1] = center[1] + margin
                polygon = create_rectangle(center, vehicle_length, vehicle_width, angle)
                polygon_points = np.array(polygon.exterior.coords, dtype=np.int32)
                cv2.polylines(frame_t, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

    if show_transform:
        x, y, w, h = margin, margin, width_t, height_t
        cv2.rectangle(frame_t, (x, y), (x + w - 2, y + h - 2), (0, 0, 255), 2)
        cv2.imshow('Transform', frame_t)
        out_t.write(frame_t)

    # 事故检测
    accident_results = accident_model(frame, verbose=False)
    accident_boxes = []
    detected_objects = False
    acc_detected = False
    involved_vehicles = set()

    accident_plot = accident_results[0].plot()
    annotated_frame = cv2.addWeighted(annotated_frame, 0.5, accident_plot, 0.5, 0)

    for result in accident_results:
        boxes_acc = result.boxes
        for box in boxes_acc:
            confidence = box.conf[0]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            accident_boxes.append((int(x1), int(y1), int(x2), int(y2), confidence.item()))
            detected_objects = True

    if detected_objects:
        if show_collision:
            collision_table = PrettyTable()
            collision_table.field_names = ["Vehicle ID", "Accident Box Confidence"]
        for acc_box in accident_boxes:
            acc_x1, acc_y1, acc_x2, acc_y2, acc_confidence = acc_box
            involved_vehicles = []

            for track_id, box in boxes.items():
                x1, y1, x2, y2 = box
                if x1 < acc_x2 and x2 > acc_x1 and y1 < acc_y2 and y2 > acc_y1:
                    involved_vehicles.append(track_id)
                    if show_collision:
                        collision_table.add_row([track_id, f"{acc_confidence: .2f}"])
            if involved_vehicles:
                acc_detected = detect_accidents(track_history, {id: boxes[id] for id in involved_vehicles},
                                                acc_confidence)
                if acc_detected:
                    for track_id in involved_vehicles:
                        x1, y1, x2, y2 = boxes[track_id]
                        overlay = annotated_frame.copy()
                        alpha = 0.5  # 半透明度
                        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 128, 255), -1)
                        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
                        cv2.putText(annotated_frame, "Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
        if show_collision:
            print(collision_table)
            print("\n")

    # 将帧写入输出视频文件
    frames.append(annotated_frame)

    # 显示带注释的帧
    resized_frame = cv2.resize(annotated_frame, (960, 540))
    cv2.imshow('Detect', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if save:
    output_frame_count = int(total_frames * (target_fps / fps))
    interpolated_frames = []

    if len(frames) < output_frame_count:
        frame_step = len(frames) / output_frame_count
        for i in range(output_frame_count):
            idx = int(i * frame_step)
            interpolated_frames.append(frames[min(idx, len(frames) - 1)])
        for frame in interpolated_frames:
            out.write(frame)
    else:
        frame_step = len(frames) / output_frame_count
        for i in range(output_frame_count):
            idx = int(i * frame_step)
            out.write(frames[min(idx, len(frames) - 1)])

end_program_time = time.time()
total_time = end_program_time - start_program_time
print(f"Total runtime: {total_time: .2f} seconds")

# 释放视频资源
cap.release()
if save:
    out.release()
if show_transform:
    out_t.release()
cv2.destroyAllWindows()
