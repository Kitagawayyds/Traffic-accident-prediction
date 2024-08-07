import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from math import radians, cos, sin

matplotlib.use('TkAgg')

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
    return rotated_points

def plot_rectangle(center, length, width, angle):
    rotated_points = create_rectangle(center, length, width, angle)
    fig, ax = plt.subplots()
    polygon = Polygon(rotated_points, closed=True, fill=None, edgecolor='r')
    ax.add_patch(polygon)
    ax.set_aspect('equal')
    ax.set_xlim(center[0] - length, center[0] + length)
    ax.set_ylim(center[1] - width, center[1] + width)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rectangle Visualization')
    plt.grid(True)
    plt.show()

# Example usage
plot_rectangle(center=(0, 0), length=10, width=5, angle=90)
