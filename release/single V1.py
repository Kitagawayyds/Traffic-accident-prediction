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

vehicle_model = YOLO("yolov10x.pt")  # 车辆模型

video_path = "accident3.mp4"  # 打开视频文件
output_path = "output.mp4"  # 输出视频路径
transform_path = 'transform.mp4'  # 映射视频路径

target_fps = 10  # 视频处理速度（帧）

risk_threshold = 6  # 风险阈值
speed_threshold = 90  # 极限速度阈值
resting_threshold = 5  # 静止阈值（消除抖动）
overlap_threshold = 0.15  # 极限重叠阈值
angle_threshold = 30  # 极限角度阈值
curvature_threshold = 0.05  # 极限曲率阈值

fluctuation_ratio = 0.5  # 波动比

box_length = 5  # 碰撞框长
box_width = 3  # 碰撞框宽

sigma = 2  # 轨迹平滑系数

# 映射区域
SOURCE = np.array([
    [1202, 424],
    [1651, 481],
    [1398, 1033],
    [531, 829]
])

TARGET = np.array([
    [0, 0],
    [15, 0],
    [15, 45],
    [0, 45]
])

scale_factor = 20  # 目标区域放大比例

margin = 50  # 留白距离

target_id = 3  # 捕获指定id

TARGET = TARGET * scale_factor
box_length = box_length * scale_factor
box_width = box_width * scale_factor

'''
假设这是视频：
+-------x-------+
|(0,0)     (x,0)|    
y               |
|(0,y)     (x,y)|
+---------------+
'''

show_matrix = False  # 打印车辆详细日志
show_track = False  # 绘制车辆轨迹
show_transform = False  # 启用映射可视化
show_region = True  # 显示源区域
save = False  # 保存推理视频
smooth = True  # 平滑轨迹
rt_display = True  # 启用imshow
capture_track = False  # 捕获指定id车辆的轨迹
traffic_counting = True  # 统计车流量

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
    ["Source Region", SOURCE],
    ["Target Region", TARGET],
    ["Scale Factor", scale_factor],
    ["Smooth Factor", sigma],
    ["Margin", margin],
    ["Target FPS", target_fps],
    ["Risk Threshold", risk_threshold],
    ["Speed Threshold", speed_threshold],
    ["Angle Threshold", angle_threshold],
    ["Fluctuation Ratio", fluctuation_ratio],
    ["Resting Threshold", resting_threshold],
    ["Overlap Threshold", overlap_threshold],
    ["Curvature Threshold", curvature_threshold],
    ["Box Length", box_length],
    ["Box Width", box_width],
    ["Capture ID", target_id],
    ["Show Detailed Metrics", bool(show_matrix)],
    ["Show Vehicle Track", bool(show_track)],
    ["Show Source Region", bool(show_region)],
    ["Transformation Visualization", bool(show_transform)],
    ["Save Output", bool(save)],
    ["Smooth track", bool(smooth)],
    ["Real-time Display", bool(rt_display)],
    ["Capture Track", bool(capture_track)],
    ["Traffic counting", bool(traffic_counting)],
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
🚥 ARMS-single V1, made by Linhan Song
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
catch_track = []


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


# 计算车辆是否在区域内
def is_vehicle_in_region(point, region):
    return cv2.pointPolygonTest(region, point, False) >= 0


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


# 平滑轨迹
def smooth_track(track):
    x_coords = [p[0] for p in track]
    y_coords = [p[1] for p in track]

    x_smooth = gaussian_filter1d(x_coords, sigma)
    y_smooth = gaussian_filter1d(y_coords, sigma)

    smoothed_track = list(zip(x_smooth, y_smooth))

    return smoothed_track


# 轨迹预处理
def get_effective_track(track_history):
    effective_track = defaultdict(lambda: deque(maxlen=30))

    for track_id, track in track_history.items():
        if smooth:
            track = smooth_track(track)
        if show_track:
            points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 255), thickness=3)
        track = view_transformer.transform_points(np.array(track))
        if smooth:
            track = smooth_track(track)
        effective_track[track_id] = track

    return effective_track


# 计算速度
def calculate_speed(track):
    velocities = np.linalg.norm(np.diff(track, axis=0), axis=1) * target_fps / scale_factor * 3.6
    avg_velocity = np.mean(velocities)
    velocity_fluctuations = np.abs(velocities - avg_velocity)
    avg_fluctuation = np.mean(velocity_fluctuations)

    return avg_velocity, avg_fluctuation


# 计算角度变化
def calculate_angle(track):
    recent_track = track[-5:]

    angles = []
    for i in range(len(recent_track) - 2):
        p1, p2, p3 = recent_track[i:i + 3]
        angle1 = atan2(p1[0] - p2[0], p1[1] - p2[1])
        angle2 = atan2(p2[0] - p3[0], p2[1] - p3[1])
        angle_change = abs(degrees(angle2 - angle1))
        angle_change = min(angle_change, 360 - angle_change)
        angles.append(angle_change)

    if not angles:
        return 0, 0

    avg_angle_change = np.mean(angles)

    last_p1, last_p2, last_p3 = recent_track[-3:]
    current_angle = degrees(atan2(last_p2[0] - last_p1[0], last_p2[1] - last_p1[1]))

    return avg_angle_change, current_angle


# 计算曲率
def calculate_curvature(track):
    def calculate(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        A = np.array([x1, y1])
        B = np.array([x2, y2])
        C = np.array([x3, y3])

        cross_product = np.linalg.norm(np.cross(B - A, C - B))

        AB = np.linalg.norm(B - A)
        BC = np.linalg.norm(C - B)

        if AB * BC == 0:
            return 0

        curvature = cross_product / (AB * BC)
        return curvature

    total_curvature = 0
    count = 0

    for i in range(1, len(track) - 1):
        p1 = track[i - 1]
        p2 = track[i]
        p3 = track[i + 1]
        total_curvature += calculate(p1, p2, p3)
        count += 1

    return total_curvature / count if count > 0 else 0


# 计算重叠度
def bb_overlap(center1, center2, angle1, angle2):
    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    max_distance = np.sqrt(box_length ** 2 + box_width ** 2)
    if distance > max_distance:
        return 0

    rect1 = create_rectangle(center1, box_length, box_width, angle1)
    rect2 = create_rectangle(center2, box_length, box_width, angle2)

    intersection_area = rect1.intersection(rect2).area

    rect_area = box_length * box_width
    overlap_ratio_1 = intersection_area / rect_area
    overlap_ratio_2 = intersection_area / rect_area

    return max(overlap_ratio_1, overlap_ratio_2)


# 计算风险评分
def calculate_risk_score(speed, fluctuation, angle, curvature, overlap):
    def speed_risk(speed):
        return min((speed / (speed_threshold * 1.3)) ** 4, 1) * 10

    def fluctuation_risk(speed, fluctuation):
        dynamic_fluctuation_threshold = max(fluctuation_ratio * speed, 20)
        return min((fluctuation / dynamic_fluctuation_threshold) ** 2, 1) * 10

    def angle_risk(angle):
        return min(((angle / angle_threshold) ** 2), 1) * 10

    def curvature_risk(speed, curvature):
        dynamic_curvature_threshold = max(speed_threshold / speed * curvature_threshold, 0.001)
        return min((curvature / dynamic_curvature_threshold) ** 2, 1) * 10

    def overlap_risk(overlap):
        return min(((overlap / overlap_threshold) ** 3), 1) * 10

    # 计算风险评分
    speed_score = speed_risk(speed)
    fluctuation_score = fluctuation_risk(speed, fluctuation)
    angle_score = angle_risk(angle)
    curvature_score = curvature_risk(speed, curvature)
    overlap_score = overlap_risk(overlap)

    # 计算最终风险评分
    max_score = max(speed_score, fluctuation_score, angle_score, curvature_score, overlap_score)
    average_score = (speed_score + fluctuation_score + angle_score + curvature_score + overlap_score) / 5
    risk_score = 0.5 * average_score + 0.5 * max_score

    return {
        'speed_score': speed_score,
        'fluctuation_score': fluctuation_score,
        'angle_score': angle_score,
        'curvature_score': curvature_score,
        'overlap_score': overlap_score,
        'risk_score': np.clip(risk_score, 0, 10)
    }


# 事故检测
def detect_accidents(track_history, boxes):
    acc_detected = False
    tracked_cars = list(track_history.keys())
    effective_track = get_effective_track(track_history)

    accident_vehicles = []
    table_m = None

    if show_matrix:
        table_m = PrettyTable()
        table_m.field_names = ["ID", "Speed", "Speed Fluctuation", "Angle Change", "Track Curvature", "Max Overlap",
                               "Speed Score", "Fluctuation Score", "Angle Score", "Curvature Score", "Overlap Score",
                               "Risk Score"]

    for track_id in tracked_cars:
        if track_id not in boxes:
            continue

        track = effective_track[track_id]
        if len(track) < 5:
            continue
        speed, fluctuation = calculate_speed(track)
        if speed < resting_threshold:
            angle_change = 0
            current_angle = 0
        else:
            angle_change, current_angle = calculate_angle(track)
        if show_transform:
            center = track[-1]
            center = (center[0] + margin, center[1] + margin)
            polygon = create_rectangle(center, box_length, box_width, current_angle)
            polygon_points = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.polylines(frame_t, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

        track_curvature = calculate_curvature(track)
        max_overlap = max(
            (bb_overlap(track[-1], effective_track[other_id][-1], current_angle,
                        calculate_angle(effective_track[other_id])[1])
             for other_id in tracked_cars if other_id != track_id and other_id in boxes), default=0)

        score = calculate_risk_score(speed, fluctuation, angle_change, track_curvature, max_overlap)

        # 解包各个分数
        speed_score = score['speed_score']
        fluctuation_score = score['fluctuation_score']
        angle_score = score['angle_score']
        curvature_score = score['curvature_score']
        overlap_score = score['overlap_score']
        risk_score = score['risk_score']

        risk_scores[track_id].append(risk_score)

        if risk_score > risk_threshold:
            acc_detected = True
            accident_vehicles.append(track_id)
            if show_matrix:
                table_m.add_row([
                    track_id,
                    f"{speed: .4f}",
                    f"{fluctuation: .4f}",
                    f"{angle_change: .4f}",
                    f"{track_curvature: .4f}",
                    f"{max_overlap: .4f}",
                    f"{speed_score: .4f}",
                    f"{fluctuation_score: .4f}",
                    f"{angle_score: .4f}",
                    f"{curvature_score: .4f}",
                    f"{overlap_score: .4f}",
                    f"{risk_score: .4f}"
                ])

    return acc_detected, accident_vehicles, table_m


start_program_time = time.time()
frame_interval = ceil(fps / target_fps)
frame_count = 0
frames = []
if traffic_counting:
    traffic_count = 0
    counted_ids = set()

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
    cv2.putText(annotated_frame, f"ARMS-single V1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (71, 210, 160), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Current Model: {vehicle_model.model_name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (71, 210, 160), 2, cv2.LINE_AA)

    # 车辆检测
    vehicle_results = vehicle_model.track(annotated_frame, persist=True, tracker='botsort.yaml', verbose=False)
    boxes = {}
    if show_region:
        cv2.fillPoly(frame, [np.array(SOURCE, dtype=np.int32)], (14, 160, 111))
    if vehicle_results[0].boxes.id is not None:
        boxes = {int(id): box.tolist() for id, box in
                 zip(vehicle_results[0].boxes.id.cpu(), vehicle_results[0].boxes.xyxy.cpu())}
        track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()
        annotated_frame = vehicle_results[0].plot()
        # 跟踪车辆
        for track_id in track_ids:
            x1, y1, x2, y2 = boxes[track_id]
            cx, cy = (x1 + x2) / 2, (y1 + 3 * y2) / 4
            if capture_track and target_id == track_id:
                catch_track.append((float(cx), float(cy)))
            track = track_history[track_id]
            track.append((float(cx), float(cy)))

            if traffic_counting and is_vehicle_in_region((cx, cy), np.array(SOURCE, dtype=np.int32)):
                if track_id not in counted_ids:
                    traffic_count += 1
                    counted_ids.add(track_id)

    # 事故检测
    acc_detected, acc_vehicles, table_m = detect_accidents(track_history, boxes)
    if traffic_counting:
        cv2.putText(annotated_frame, f"Traffic Count: {traffic_count}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if show_transform:
        x, y, w, h = margin, margin, width_t, height_t
        cv2.rectangle(frame_t, (x, y), (x + w - 2, y + h - 2), (0, 0, 255), 2)
        cv2.imshow('Transform', frame_t)
        out_t.write(frame_t)
    if acc_detected:
        for track_id in acc_vehicles:
            x1, y1, x2, y2 = boxes[track_id]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 128, 255), -1)
            cv2.putText(annotated_frame, "Accident Detected!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

    annotated_frame = cv2.addWeighted(annotated_frame, 0.5, frame, 0.5, 0)

    if show_matrix and table_m and len(table_m.rows) > 0:
        print("\n")
        print(table_m)
        print("\n")

    # 将帧写入输出视频文件
    frames.append(annotated_frame)

    # 显示带注释的帧
    if rt_display:
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
if capture_track:
    print(f"Capture Track: {catch_track}")
if save:
    out.release()
if show_transform:
    out_t.release()
cv2.destroyAllWindows()
