import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as ShapelyPolygon

matplotlib.use('TkAgg')


def draw_bbox(ax, bbox, color, label):
    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                         linewidth=2, edgecolor=color, facecolor='none', label=label)
    ax.add_patch(rect)


def visualize_overlap(bbox1, bbox2):
    fig, ax = plt.subplots()

    # Draw bounding boxes
    draw_bbox(ax, bbox1, 'blue', 'BBox 1')
    draw_bbox(ax, bbox2, 'red', 'BBox 2')

    # Convert to Shapely Polygons
    poly1 = ShapelyPolygon([(bbox1[0], bbox1[1]), (bbox1[0], bbox1[3]), (bbox1[2], bbox1[3]), (bbox1[2], bbox1[1])])
    poly2 = ShapelyPolygon([(bbox2[0], bbox2[1]), (bbox2[0], bbox2[3]), (bbox2[2], bbox2[3]), (bbox2[2], bbox2[1])])

    # Calculate intersection and union
    intersection = poly1.intersection(poly2)
    union = poly1.union(poly2)

    # Draw intersection if it exists
    if not intersection.is_empty:
        x, y = intersection.exterior.xy
        ax.plot(x, y, color='green', linestyle='--', linewidth=2, label='Intersection')
        ax.fill(x, y, color='lightgreen', alpha=0.5, label='Intersection Area')

    # Draw polygons
    x, y = poly1.exterior.xy
    ax.plot(x, y, color='blue', linestyle='-', linewidth=2, label='BBox 1')

    x, y = poly2.exterior.xy
    ax.plot(x, y, color='red', linestyle='-', linewidth=2, label='BBox 2')

    ax.set_xlim(min(bbox1[0], bbox2[0]) - 10, max(bbox1[2], bbox2[2]) + 10)
    ax.set_ylim(min(bbox1[1], bbox2[1]) - 10, max(bbox1[3], bbox2[3]) + 10)
    ax.set_aspect('equal', 'box')
    ax.legend()
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Bounding Box Overlap Visualization')

    # Calculate traditional overlap ratio
    intersection_area = intersection.area
    union_area = union.area
    traditional_overlap_ratio = intersection_area / union_area

    # Calculate overlap ratios from both perspectives
    overlap_ratio_1 = intersection_area / poly1.area
    overlap_ratio_2 = intersection_area / poly2.area
    perspective_overlap_ratio = max(overlap_ratio_1, overlap_ratio_2)

    # Display overlap ratios
    plt.figtext(0.15, 0.2, f'Traditional Overlap Ratio: {traditional_overlap_ratio:.2f}', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))
    plt.figtext(0.15, 0.1, f'Perspective Overlap Ratio: {perspective_overlap_ratio:.2f}', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))

    plt.grid(True)
    plt.show()


# 示例边界框
bbox1 = [4, 4, 6, 6]
bbox2 = [4, 4, 8, 8]

visualize_overlap(bbox1, bbox2)
