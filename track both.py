from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO, solutions

# 加载YOLOv8模型
model = YOLO("yolov10n.pt")
names = model.model.names

# 打开视频文件
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

w, h, fps = (int(cap.get(x)) for x in (
cv2.CAP_PROP_FRAME_WIDTH,
cv2.CAP_PROP_FRAME_HEIGHT,
cv2.CAP_PROP_FPS))

print(w,h,fps)

line_pts = [(0, 3/4*h), (w, 3/4*h)]

speed_obj = solutions.SpeedEstimator(
                                     names=names,
                                     view_img=True,
                                     line_thickness=3
                                     )

# 存储轨迹历史
track_history = defaultdict(lambda: [])

# 循环读取视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在帧上运行YOLOv8跟踪，并在帧之间保留跟踪信息
        results = model.track(frame, persist=True, tracker='bytetrack.yaml')
        frame = speed_obj.estimate_speed(frame, results)

        # 检查是否有检测到的框
        if results[0].boxes.id is not None:
            # 获取框和跟踪ID
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # 在帧上可视化结果
            annotated_frame = results[0].plot()

            # 绘制跟踪线
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y 中心点
                if len(track) > 30:  # 保留30帧的轨迹
                    track.pop(0)

                # 画出跟踪线
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # 显示带注释的帧
            cv2.imshow("Tracking", annotated_frame)

        # 如果按下'q'键则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则退出循环
        break

# 释放视频捕捉对象
cap.release()
cv2.destroyAllWindows()

# 将最终轨迹信息保存到txt文件
with open("track_history.txt", "w") as f:
    for track_id, track in track_history.items():
        f.write(f"ID: {track_id}, Track: {track}\n")
