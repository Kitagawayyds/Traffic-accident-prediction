import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

matplotlib.use('TkAgg')


def ellipse_area(a, b):
    """计算椭圆的面积"""
    return np.pi * a * b


def ellipse_intersection_area(e1, e2):
    """计算两个椭圆的交集面积"""
    e1_polygon = e1
    e2_polygon = e2
    intersection_area = e1_polygon.intersection(e2_polygon).area
    return intersection_area


def create_ellipse(center, a, b):
    """创建椭圆对象，椭圆的长轴为a，短轴为b"""
    angle = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + a * np.cos(angle)
    y = center[1] + b * np.sin(angle)
    points = np.vstack((x, y)).T
    return Polygon(points)


def bb_overlap_ellipse(bbox1, bbox2):
    """计算两个椭圆形状的重叠度"""
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    a1 = (bbox1[2] - bbox1[0]) / 2
    b1 = (bbox1[3] - bbox1[1]) / 2

    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    a2 = (bbox2[2] - bbox2[0]) / 2
    b2 = (bbox2[3] - bbox2[1]) / 2

    ellipse1 = create_ellipse(center1, a1, b1)
    ellipse2 = create_ellipse(center2, a2, b2)

    intersection_area = ellipse_intersection_area(ellipse1, ellipse2)

    area1 = ellipse_area(a1, b1)
    area2 = ellipse_area(a2, b2)

    overlap_ratio_1 = intersection_area / area1
    overlap_ratio_2 = intersection_area / area2
    return max(overlap_ratio_1, overlap_ratio_2), ellipse1, ellipse2, intersection_area


def plot_ellipses(bbox1, bbox2):
    """绘制椭圆、重叠区域、重叠率及边界框"""
    overlap_ratio, ellipse1, ellipse2, intersection_area = bb_overlap_ellipse(bbox1, bbox2)

    fig, ax = plt.subplots()

    # 绘制边界框
    rect1 = plt.Rectangle((bbox1[0], bbox1[1]), bbox1[2] - bbox1[0], bbox1[3] - bbox1[1],
                          edgecolor='blue', facecolor='none', linestyle='--', label='Bounding Box 1', linewidth=2)
    rect2 = plt.Rectangle((bbox2[0], bbox2[1]), bbox2[2] - bbox2[0], bbox2[3] - bbox2[1],
                          edgecolor='red', facecolor='none', linestyle='--', label='Bounding Box 2', linewidth=2)
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    # 绘制椭圆
    x, y = ellipse1.exterior.xy
    ax.plot(x, y, label='Ellipse 1', color='blue')

    x, y = ellipse2.exterior.xy
    ax.plot(x, y, label='Ellipse 2', color='red')

    # 绘制重叠区域
    overlap_polygon = ellipse1.intersection(ellipse2)
    if not overlap_polygon.is_empty:
        x, y = overlap_polygon.exterior.xy
        ax.fill(x, y, color='purple', alpha=0.5, label='Intersection')

    # 设置图例和标签
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(f'Overlap Ratio: {overlap_ratio:.2f}')
    plt.grid(True)

    plt.show()


# Example bounding boxes
bbox1 = (2, 1, 4, 5)
bbox2 = (3, 3, 7, 5)

plot_ellipses(bbox1, bbox2)
