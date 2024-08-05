import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from math import radians, cos, sin

matplotlib.use('TkAgg')

def create_rectangle(center, width, height, angle):
    cx, cy = center
    angle = radians(angle)
    rect_points = np.array([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2]
    ])
    rotation_matrix = np.array([
        [cos(angle), -sin(angle)],
        [sin(angle), cos(angle)]
    ])
    rotated_points = np.dot(rect_points, rotation_matrix)
    rotated_points[:, 0] += cx
    rotated_points[:, 1] += cy
    return Polygon(rotated_points)


def bb_overlap(bbox1, bbox2, angle1, angle2, width, height):
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)

    rect1 = create_rectangle(center1, width, height, angle1)
    rect2 = create_rectangle(center2, width, height, angle2)

    intersection_area = rect1.intersection(rect2).area

    rect_area = width * height
    overlap_ratio_1 = intersection_area / rect_area
    overlap_ratio_2 = intersection_area / rect_area

    # Visualization
    fig, ax = plt.subplots()

    # Plot the rectangles
    x1, y1 = rect1.exterior.xy
    x2, y2 = rect2.exterior.xy
    ax.plot(x1, y1, label='Rectangle 1', color='blue')
    ax.plot(x2, y2, label='Rectangle 2', color='red')

    # Plot the intersection area
    intersection = rect1.intersection(rect2)
    if intersection.is_empty:
        print("No intersection found.")
    else:
        x_int, y_int = intersection.exterior.xy
        ax.fill(x_int, y_int, color='purple', alpha=0.5, label='Intersection')

    ax.set_aspect('equal')
    plt.legend()
    plt.title(f'Overlap Ratio: {max(overlap_ratio_1, overlap_ratio_2):.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

    return max(overlap_ratio_1, overlap_ratio_2)


# Example usage
bbox1 = (10, 10, 50, 50)
bbox2 = (50, 50, 90, 90)
angle1 = 30
angle2 = 60
width = 40
height = 80

overlap = bb_overlap(bbox1, bbox2, angle1, angle2, width, height)
print(f'Overlap Ratio: {overlap:.2f}')
