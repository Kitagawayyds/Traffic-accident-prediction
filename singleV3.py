import argparse
import os
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


# 读取映射文件
def load_data_from_txt(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    scale_factor = int(lines[0].strip())
    source_lines = lines[1:5]
    source = np.array([list(map(int, line.strip().strip('[]').split(','))) for line in source_lines])
    target_lines = lines[5:9]
    target = np.array([list(map(int, line.strip().strip('[]').split(','))) for line in target_lines])

    return scale_factor, source, target


# 获取车辆参数
def load_vehicle_specs(file_path, scale_factor):
    vehicle_specs = {}
    default_specs = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.strip().split(':')
                if len(parts) == 2:
                    vehicle_type, dimensions = parts
                    length, width = map(float, dimensions.split(','))
                    if vehicle_type.lower() == 'else':
                        default_specs = (length * scale_factor, width * scale_factor)
                    else:
                        vehicle_specs[vehicle_type] = (length * scale_factor, width * scale_factor)
    return vehicle_specs, default_specs


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


# 计算车辆是否在区域内
def is_vehicle_in_region(point, region):
    return cv2.pointPolygonTest(region, point, False) >= 0


def parse_arguments():
    parser = argparse.ArgumentParser(description="ARMS")

    parser.add_argument('--vehicle_model', type=str, default="yolov10n.pt", help="Path to the vehicle detection model")
    parser.add_argument('--video_path', type=str, default="test2.mp4", help="Path to the input video file")
    parser.add_argument('--output_path', type=str, default="output.mp4", help="Path to the output video file")
    parser.add_argument('--transform_path', type=str, default='transform.mp4', help="Path to the transformation video")
    parser.add_argument('--transform_file_path', type=str, default='transform.txt', help="Path to the transformation file")
    parser.add_argument('--spec_file_path', type=str, default='spec.txt', help="Path to the vehicle spec file")
    '''
    {scale_factor}
    四个数组（源）
    四个数组（目标）
    假设这是视频：
    +-------x-------+
    |(0,0)     (x,0)|    
    y               |
    |(0,y)     (x,y)|
    +---------------+
    '''
    parser.add_argument('--target_fps', type=int, default=20, help="Frame rate for video processing")

    parser.add_argument('--risk_threshold', type=float, default=6, help="Risk threshold value")
    parser.add_argument('--speed_threshold', type=float, default=100, help="Speed threshold value")
    parser.add_argument('--resting_threshold', type=int, default=5, help="Resting threshold to eliminate jitter")
    parser.add_argument('--overlap_threshold', type=float, default=0.15, help="Maximum overlap threshold")
    parser.add_argument('--angle_threshold', type=float, default=30, help="Maximum angle threshold")
    parser.add_argument('--curvature_threshold', type=float, default=0.05, help="Maximum curvature threshold")

    parser.add_argument('--fluctuation_ratio', type=float, default=0.5, help="Fluctuation ratio")
    parser.add_argument('--sigma', type=float, default=2, help="Smoothing coefficient for trajectory")
    parser.add_argument('--margin', type=int, default=200, help="Margin distance")
    parser.add_argument('--target_id', type=int, default=9, help="ID of the vehicle to capture")

    parser.add_argument('--show_matrix', type=bool, default=True, help="Print detailed vehicle logs")
    parser.add_argument('--show_track', type=bool, default=True, help="Draw vehicle trajectories")
    parser.add_argument('--show_transform', type=bool, default=True, help="Enable transformation visualization")
    parser.add_argument('--show_region', type=bool, default=True, help="Show source region")
    parser.add_argument('--save', type=bool, default=True, help="Save inference video")
    parser.add_argument('--smooth', type=bool, default=True, help="Smooth trajectories")
    parser.add_argument('--rt_display', type=bool, default=True, help="Enable imshow")
    parser.add_argument('--capture_track', type=bool, default=False, help="Capture trajectory of the specified vehicle ID")
    parser.add_argument('--traffic_counting', type=bool, default=True, help="Count traffic volume")

    return parser.parse_args()


def validate_parameters(args, video_fps):
    if args.target_fps <= 0 or args.target_fps > video_fps:
        raise ValueError("Target FPS must be a positive integer and less than or equal to the video's FPS.")
    if args.risk_threshold <= 0 or args.risk_threshold > 10:
        raise ValueError("Risk threshold must be a positive float and less than or equal to 10.")
    if args.speed_threshold <= 0:
        raise ValueError("Speed threshold must be a positive float.")
    if args.resting_threshold < 0:
        raise ValueError("Resting threshold must be a non-negative integer.")
    if not (0 <= args.overlap_threshold <= 1):
        raise ValueError("Overlap threshold must be between 0 and 1.")
    if args.angle_threshold < 0 or args.angle_threshold > 180:
        raise ValueError("Angle threshold must be a non-negative float and less than or equal to 180.")
    if args.curvature_threshold <= 0:
        raise ValueError("Curvature threshold must be a positive float.")
    if args.fluctuation_ratio < 0:
        raise ValueError("Fluctuation ratio must be a non-negative float.")
    if args.sigma <= 0:
        raise ValueError("Smoothing coefficient (sigma) must be a positive float.")
    if args.margin < 0:
        raise ValueError("Margin distance must be a non-negative integer.")
    if args.target_id < 0:
        raise ValueError("Target ID must be a non-negative integer.")


# 检查文件路径
def check_file_path(file_path, file_type):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_type} file not found: {file_path}")


# 初始化
def initializer():
    try:
        args = parse_arguments()
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        raise

    try:
        check_file_path(args.vehicle_model, "Vehicle model")
        vehicle_model = YOLO(args.vehicle_model)
    except Exception as e:
        print(f"Error loading vehicle model: {e}")
        raise

    video_path = args.video_path
    output_path = args.output_path
    transform_path = args.transform_path
    transform_file_path = args.transform_file_path
    spec_file_path = args.spec_file_path

    try:
        check_file_path(video_path, "Video")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file at path: {video_path}")

        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        validate_parameters(args, video_fps)
    except Exception as e:
        print(f"Error opening video file or validating parameters: {e}")
        raise

    try:
        check_file_path(transform_file_path, "Transformation file")
        scale_factor, SOURCE, TARGET = load_data_from_txt(transform_file_path)
    except Exception as e:
        print(f"Error loading data from file: {e}")
        raise

    try:
        check_file_path(spec_file_path, "Vehicle Spec file")
        vehicle_specs, default_specs = load_vehicle_specs(spec_file_path, scale_factor)
    except Exception as e:
        print(f"Error loading data from file: {e}")
        raise

    try:
        target_fps = args.target_fps
        risk_threshold = args.risk_threshold
        speed_threshold = args.speed_threshold
        resting_threshold = args.resting_threshold
        overlap_threshold = args.overlap_threshold
        angle_threshold = args.angle_threshold
        curvature_threshold = args.curvature_threshold
        fluctuation_ratio = args.fluctuation_ratio
        sigma = args.sigma
        margin = args.margin
        target_id = args.target_id
        show_matrix = args.show_matrix
        show_track = args.show_track
        show_transform = args.show_transform
        show_region = args.show_region
        save = args.save
        smooth = args.smooth
        rt_display = args.rt_display
        capture_track = args.capture_track
        traffic_counting = args.traffic_counting

        TARGET = TARGET * scale_factor

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width_t = np.abs(TARGET[1, 0] - TARGET[0, 0])
        height_t = np.abs(TARGET[2, 1] - TARGET[1, 1])
    except Exception as e:
        print(f"Error processing video or parameters: {e}")
        raise

    try:
        vehicle_specs_list = []
        for vehicle_type, (length, width) in vehicle_specs.items():
            vehicle_specs_list.append([vehicle_type, length, width])
        if default_specs:
            vehicle_specs_list.append(["Default", default_specs[0], default_specs[1]])

        # 更新配置数据
        config_data = [
            ["Parameter", "Value"],
            ["Video Path", video_path],
            ["Output Video Path", output_path],
            ["Transform Video Path", transform_path],
            ["Transform File Path", transform_file_path],
            ["Vehicle Model", args.vehicle_model],
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

        # 打印车辆配置数据
        vehicle_specs_table = PrettyTable()
        vehicle_specs_table.field_names = ["Vehicle Type", "Length", "Width"]
        for row in vehicle_specs_list:
            vehicle_specs_table.add_row(row)

        # 打印其他配置数据
        config_table = PrettyTable()
        config_table.field_names = config_data[0]
        for row in config_data[1:]:
            config_table.add_row(row)
        print("Config Configuration: ")
        print(config_table)
        print("Vehicle Specs: ")
        print(vehicle_specs_table)
    except Exception as e:
        print(f"Error creating or printing the configuration table: {e}")
        raise

    return (vehicle_model, video_path, output_path, transform_path, transform_file_path,
            target_fps, risk_threshold, speed_threshold, resting_threshold, overlap_threshold,
            angle_threshold, curvature_threshold, fluctuation_ratio, default_specs, vehicle_specs,
            sigma, margin, target_id, show_matrix, show_track, show_transform, show_region,
            save, smooth, rt_display, capture_track, traffic_counting, cap, scale_factor,
            SOURCE, TARGET, frame_width, frame_height, fps, total_frames, width_t, height_t)


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


# 程序起点
start_program_time = time.time()
software_info = """
🚥 ARMS-single V2, made by Linhan Song
If you want to get the latest version and the detailed information of the program, please visit:
https://github.com/Kitagawayyds/Traffic-accident-prediction
"""
print(software_info)

# 初始化
(vehicle_model, video_path, output_path, transform_path, transform_file_path,
 target_fps, risk_threshold, speed_threshold, resting_threshold, overlap_threshold,
 angle_threshold, curvature_threshold, fluctuation_ratio, default_specs, vehicle_specs,
 sigma, margin, target_id, show_matrix, show_track, show_transform, show_region,
 save, smooth, rt_display, capture_track, traffic_counting, cap, scale_factor,
 SOURCE, TARGET, frame_width, frame_height, fps, total_frames, width_t, height_t) = initializer()

model_table = PrettyTable()
model_table.field_names = ["Class ID", "Class Name"]
for id, name in vehicle_model.names.items():
    model_table.add_row([id, name])
print(f"{vehicle_model.model_name} Class Details: ")
print(model_table)

# 创建视频写入对象
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
if save:
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))
if show_transform:
    out_t = cv2.VideoWriter(transform_path, fourcc, target_fps, (width_t + margin * 2, height_t + margin * 2))

# 存储跟踪历史记录、风险评分、事故置信度等
track_history = defaultdict(lambda: deque(maxlen=30))
risk_scores = defaultdict(lambda: deque(maxlen=30))
last_known_angles = defaultdict(lambda: deque(maxlen=200))
accident_confidences = deque(maxlen=30)
catch_track = []

view_transformer = ViewTransformer(source=SOURCE, target=TARGET)


# 平滑轨迹
def smooth_track(track):
    x_coords = [p[0] for p in track]
    y_coords = [p[1] for p in track]

    x_smooth = gaussian_filter1d(x_coords, sigma)
    y_smooth = gaussian_filter1d(y_coords, sigma)

    smoothed_track = list(zip(x_smooth, y_smooth))

    return smoothed_track


# 轨迹预处理
def get_effective_track(track_history, boxes):
    effective_track = defaultdict(lambda: deque(maxlen=30))

    for track_id, track in track_history.items():
        if track_id not in boxes:
            continue
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


# 计算重叠度
def bb_overlap(center1, center2, angle1, angle2, class1, class2):
    length1, width1 = vehicle_specs.get(class1, default_specs)
    length2, width2 = vehicle_specs.get(class2, default_specs)

    distance = np.linalg.norm(np.array(center1) - np.array(center2))

    max_distance = np.sqrt(max(length1, length2) ** 2 + max(width1, width2) ** 2)
    if distance > max_distance:
        return 0

    rect1 = create_rectangle(center1, length1, width1, angle1)
    rect2 = create_rectangle(center2, length2, width2, angle2)

    intersection_area = rect1.intersection(rect2).area

    rect_area1 = length1 * width1
    rect_area2 = length2 * width2
    overlap_ratio_1 = intersection_area / rect_area1
    overlap_ratio_2 = intersection_area / rect_area2

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
def detect_accidents(track_history, boxes, class_name):
    acc_detected = False
    tracked_cars = list(track_history.keys())
    effective_track = get_effective_track(track_history, boxes)

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
            current_angle = last_known_angles.get(track_id, 0)
        else:
            angle_change, current_angle = calculate_angle(track)
            last_known_angles[track_id] = current_angle

        track_curvature = calculate_curvature(track)
        max_overlap = max(
            (
                bb_overlap(
                    track[-1],
                    effective_track[other_id][-1],
                    current_angle,
                    calculate_angle(effective_track[other_id])[1] if calculate_speed(effective_track[other_id])[0] >= resting_threshold else last_known_angles.get(other_id, 0),
                    class_name[track_id],
                    class_name[other_id]
                )
                for other_id in tracked_cars
                if other_id != track_id and other_id in boxes and len(effective_track[other_id]) >= 5
            ),
            default=0
        )

        score = calculate_risk_score(speed, fluctuation, angle_change, track_curvature, max_overlap)

        # 解包各个分数
        speed_score = score['speed_score']
        fluctuation_score = score['fluctuation_score']
        angle_score = score['angle_score']
        curvature_score = score['curvature_score']
        overlap_score = score['overlap_score']
        risk_score = score['risk_score']

        risk_scores[track_id].append(risk_score)

        if show_transform:
            center = track[-1]
            center = (center[0] + margin, center[1] + margin)
            vehicle_type = class_name[track_id]
            length, width = vehicle_specs.get(vehicle_type, default_specs)
            polygon = create_rectangle(center, length, width, current_angle)
            polygon_points = np.array(polygon.exterior.coords, dtype=np.int32)
            margin_array = np.full_like(track, margin)
            track = track + margin_array
            points = np.array(track).astype(np.int32).reshape((-1, 1, 2))

        if risk_score > risk_threshold:
            acc_detected = True
            accident_vehicles.append(track_id)
            if show_transform:
                cv2.fillPoly(frame_t, [polygon_points], color=(0, 0, 255))
                cv2.putText(frame_t, "Accident Detected!", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 147, 0), 1)
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
        if show_transform:
            cv2.polylines(frame_t, [points], isClosed=False, color=(0, 255, 109), thickness=2)
            cv2.polylines(frame_t, [polygon_points], isClosed=True, color=(0, 200, 0), thickness=2)
            text_lines = [f"ID: {track_id}", f"Class: {vehicle_type}", f"Risk Score: {risk_score: .2f}"]
            y_offset = -20
            for line in text_lines:
                text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_width, text_height = text_size
                text_x = int(center[0] - text_width / 2)
                text_y = int(center[1] + text_height / 2 + y_offset)
                cv2.putText(frame_t, line, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (129, 47, 51), 1, cv2.LINE_AA)
                y_offset += text_height + 5
    return acc_detected, accident_vehicles, table_m


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
    cv2.putText(annotated_frame, f"ARMS-single V2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (71, 210, 160), 2,
                cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Current Model: {vehicle_model.model_name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (71, 210, 160), 2, cv2.LINE_AA)

    # 车辆检测
    vehicle_results = vehicle_model.track(annotated_frame, persist=True, tracker='botsort.yaml', verbose=False)
    boxes = {}
    class_names = {}
    if show_region:
        cv2.fillPoly(frame, [np.array(SOURCE, dtype=np.int32)], (14, 160, 111))
    if vehicle_results[0].boxes.id is not None:
        boxes = {int(id): box.tolist() for id, box in
                 zip(vehicle_results[0].boxes.id.cpu(), vehicle_results[0].boxes.xyxy.cpu())}
        track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()
        cls_list = vehicle_results[0].boxes.cls.cpu().tolist()
        class_names = {id: vehicle_results[0].names[int(cls)] for id, cls in zip(track_ids, cls_list)}
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
    acc_detected, acc_vehicles, table_m = detect_accidents(track_history, boxes, class_names)
    if traffic_counting:
        cv2.putText(annotated_frame, f"Traffic Count: {traffic_count}", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if show_transform:
        x, y, w, h = margin, margin, width_t, height_t
        cv2.rectangle(frame_t, (x, y), (x + w - 2, y + h - 2), (129, 47, 51), 2)
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

# 释放视频资源
cap.release()
if capture_track:
    print(f"Capture Track: {catch_track}")
if save:
    out.release()
if show_transform:
    out_t.release()
cv2.destroyAllWindows()

end_program_time = time.time()
total_time = end_program_time - start_program_time
print(f"Total runtime: {total_time: .2f} seconds")