import cv2
from ultralytics import YOLO

def predict_video(video_path, model_path):
    # 加载 YOLO 模型
    model = YOLO(model_path)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # 逐帧读取视频
        ret, frame = cap.read()
        if not ret:
            break

        # 进行预测
        results = model(frame)

        # 输出每个检测的置信度
        for result in results:
            print("Detected objects in frame:")
            # 获取检测框的置信度和类别
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0]  # 置信度
                class_id = int(box.cls[0])  # 类别ID
                class_name = model.names[class_id]  # 类别名称
                print(f"Class: {class_name}, Confidence: {confidence:.2f}")

        # 显示带有预测结果的帧（可选）
        # cv2.imshow('Frame', results.render()[0])
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 设定视频路径和模型路径
    video_path = 'accident.mp4'
    model_path = 'yolov10n.pt'  # 使用 YOLOv8 模型路径
    predict_video(video_path, model_path)
