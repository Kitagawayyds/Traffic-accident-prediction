import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

matplotlib.use('TkAgg')

# Define the ViewTransformer class
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

# Example source and target points for perspective transformation
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

scale_factor = 20  # 目标区域放大比例

TARGET = TARGET * scale_factor

# Initialize ViewTransformer
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

def smooth_track(track, sigma):
    if len(track) < 3:
        return track

    x_coords = [p[0] for p in track]
    y_coords = [p[1] for p in track]

    x_smooth = gaussian_filter1d(x_coords, sigma)
    y_smooth = gaussian_filter1d(y_coords, sigma)

    smoothed_track = list(zip(x_smooth, y_smooth))

    return smoothed_track

def plot_smoothed_and_transformed_tracks(smoothed_track, transformed_track):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'wspace': 0.05})

    # Determine the limits for the x and y axes based on SOURCE and TARGET
    x_limits = [SOURCE[0, 0], SOURCE[2, 0]]
    y_limits = [SOURCE[0, 1], SOURCE[2, 1]]

    # Plot smoothed track
    smoothed_x = [p[0] for p in smoothed_track]
    smoothed_y = [p[1] for p in smoothed_track]

    axs[0].plot(smoothed_x, smoothed_y, 'x-', label='Smoothed Track', color='red')

    # Annotate points with their indices
    for i, (x, y) in enumerate(smoothed_track):
        axs[0].annotate(i, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')

    axs[0].set_xlim(x_limits)
    axs[0].set_ylim(y_limits)
    axs[0].set_xlabel('X Coordinate (pixels)')
    axs[0].set_ylabel('Y Coordinate (pixels)')
    axs[0].set_title('Smoothed Track')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].invert_yaxis()
    axs[0].set_aspect('auto')  # Adjust aspect ratio to fill the space

    # Plot transformed track
    transformed_x = [p[0] for p in transformed_track]
    transformed_y = [p[1] for p in transformed_track]

    axs[1].plot(transformed_x, transformed_y, 's-', label='Transformed Track', color='green')

    # Annotate points with their indices
    for i, (x, y) in enumerate(transformed_track):
        axs[1].annotate(i, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='green')

    axs[1].set_xlim(TARGET[:, 0].min(), TARGET[:, 0].max())
    axs[1].set_ylim(TARGET[:, 1].min(), TARGET[:, 1].max())
    axs[1].set_xlabel('X Coordinate (pixels)')
    axs[1].set_ylabel('Y Coordinate (pixels)')
    axs[1].set_title('Transformed Track')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].invert_yaxis()
    axs[1].set_aspect('auto')  # Adjust aspect ratio to fill the space

    plt.tight_layout()
    plt.show()

# Example usage
track = [(1019.8986969816142, 1036.9507009589852), (1013.6781092748033, 1031.8394026047642), (1002.5160158723779, 1022.4215572497509), (988.1869474149373, 1009.5936123658246), (972.1960737407298, 993.5043600953177), (955.578147485304, 973.3195374263572), (939.0494858311474, 947.8651728123054), (923.12110478901, 916.7036573369419), (908.1237578710658, 880.8073229001109), (894.252807401944, 842.29266815369), (881.6118073844798, 803.4755171181948), (870.2148357244719, 766.0459753089025), (859.9895863021845, 730.8647765958508), (850.8153427030776, 698.2288028810792), (842.591881698316, 668.269989750075), (835.3232119318195, 641.3556060802739), (829.1974692236938, 618.4483676521799), (824.6231958243615, 601.2655393656003), (822.1363504255218, 591.9137465151832)]
sigma = 1.0

smoothed_track = smooth_track(track, sigma)
transformed_track = view_transformer.transform_points(np.array(track))

# Plot the smoothed and transformed tracks side-by-side
plot_smoothed_and_transformed_tracks(smoothed_track, transformed_track)
