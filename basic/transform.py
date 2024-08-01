import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')

# 定义视角转换类
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

# 更新源和目标区域
SOURCE = np.array([
    [300, 150],  # 左上角
    [500, 150],  # 右上角
    [600, 550],  # 右下角
    [200, 550]   # 左下角
])

TARGET = np.array([
    [125, 50],   # 左上角
    [275, 50],   # 右上角
    [275, 350],  # 右下角
    [125, 350]   # 左下角
])

# 初始化视角转换器
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# 示例图像和车辆边界框
image_width, image_height = 800, 800
image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255  # 白色背景

# 画一个示例车辆边界框
bbox = [350, 300, 450, 350]  # [x1, y1, x2, y2]
cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

# 计算车辆中心点
x1, y1, x2, y2 = bbox
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
vehicle_point = np.array([[cx, cy]])

# 转换车辆坐标
transformed_point = view_transformer.transform_points(vehicle_point)

# 画源区域
for i in range(len(SOURCE)):
    cv2.line(image, tuple(SOURCE[i]), tuple(SOURCE[(i + 1) % len(SOURCE)]), (255, 0, 0), 2)  # 红色线条，线宽2

# 画转化前的车辆中心点
image_with_bbox = image.copy()
cv2.circle(image_with_bbox, (int(cx), int(cy)), 10, (0, 0, 255), -1)  # 红色圆点
cv2.putText(image_with_bbox, f"Vehicle: ({int(cx)}, {int(cy)})", (int(cx), int(cy) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# 创建目标图像
target_width, target_height = 400, 400
image_with_transformed_point = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255  # 白色背景

# 画目标区域
for i in range(len(TARGET)):
    cv2.line(image_with_transformed_point, tuple(TARGET[i]), tuple(TARGET[(i + 1) % len(TARGET)]), (0, 0, 255), 2)  # 蓝色线条，线宽2

# 画转换后的车辆中心点
transformed_x, transformed_y = transformed_point[0]
transformed_x, transformed_y = int(transformed_x), int(transformed_y)  # 去掉偏移调整
cv2.circle(image_with_transformed_point, (transformed_x, transformed_y), 10, (0, 255, 0), -1)  # 绿色圆点
cv2.putText(image_with_transformed_point, f"Vehicle: ({int(transformed_x)}, {int(transformed_y)})", (transformed_x, transformed_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# 添加坐标文本
for pt in SOURCE:
    cv2.putText(image_with_bbox, f"{tuple(pt)}", (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
for pt in TARGET:
    cv2.putText(image_with_transformed_point, f"{tuple(pt)}", (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# 显示图像
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 转化前
axes[0].imshow(cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB))
axes[0].set_title('Before Transformation')
axes[0].axis('off')

# 转化后
axes[1].imshow(cv2.cvtColor(image_with_transformed_point, cv2.COLOR_BGR2RGB))
axes[1].set_title('After Transformation')
axes[1].axis('off')

plt.show()