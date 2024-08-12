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
track = [(1162.8311157226562, 361.63380432128906), (1155.080322265625, 362.50572204589844), (1147.0523071289062, 368.81736755371094), (1137.1571044921875, 374.02042388916016), (1134.3131713867188, 375.8381881713867), (1129.132080078125, 379.2957229614258), (1121.828857421875, 382.2553939819336), (1114.6808471679688, 387.2833557128906), (1107.5, 393.73644256591797), (1103.0662841796875, 405.22113037109375), (1101.9371337890625, 413.1311569213867), (1096.2974853515625, 416.7002868652344), (1080.0239868164062, 407.5585479736328), (1070.6035766601562, 404.7084732055664), (1056.5574951171875, 405.1902542114258), (1045.3810424804688, 409.9651336669922), (1041.7229614257812, 411.5669708251953), (1033.6058349609375, 416.26600646972656), (1027.0385131835938, 419.1797180175781), (1016.2973327636719, 422.9189224243164), (1008.2904663085938, 427.72825622558594), (994.9538879394531, 431.61121368408203), (990.0263671875, 432.93084716796875), (980.7971801757812, 439.3899688720703), (970.19677734375, 444.7046432495117), (959.6625366210938, 451.4016342163086), (950.5552368164062, 456.9991760253906), (935.8836669921875, 462.73802947998047), (930.5816955566406, 464.7231140136719), (919.4684448242188, 470.15604400634766), (911.2306213378906, 475.5327911376953), (904.28466796875, 480.6148910522461), (896.5807495117188, 483.60604095458984), (879.8513488769531, 489.5604782104492), (873.90234375, 491.5414581298828), (857.5907287597656, 501.9646682739258), (844.2752990722656, 511.13407135009766), (831.6216125488281, 522.9827575683594), (747.1840209960938, 551.0816040039062), (732.2169189453125, 560.3010101318359), (714.1466064453125, 570.1258010864258), (707.4580383300781, 573.7412567138672), (693.2084350585938, 580.3426208496094), (675.2582397460938, 590.2719268798828), (656.2352905273438, 597.9717407226562), (635.7272338867188, 607.2695007324219), (613.5559387207031, 617.3661193847656), (605.3328552246094, 621.1074676513672), (577.2151641845703, 632.6067047119141), (564.0766296386719, 640.3760681152344), (546.3350677490234, 651.767578125), (524.8279113769531, 661.7804565429688), (498.2327423095703, 670.9687042236328), (488.3606719970703, 674.2800445556641), (468.8695373535156, 684.5726623535156), (447.65802001953125, 698.2506408691406), (422.5874328613281, 710.6771697998047), (395.69866943359375, 728.2183227539062), (366.4038543701172, 742.6362762451172), (355.5034637451172, 747.6658935546875), (332.06065368652344, 759.1273040771484), (303.12255096435547, 773.7040863037109), (272.8040084838867, 789.7104034423828), (239.71170806884766, 805.6217803955078), (205.14176559448242, 825.5965881347656), (192.23122787475586, 833.1271057128906), (164.2571792602539, 846.3757476806641), (133.8844118118286, 860.9453277587891), (115.52433013916016, 877.2683258056641), (97.32510375976562, 883.3189697265625), (78.21692657470703, 887.5542602539062), (72.56782531738281, 888.1125640869141), (57.00889205932617, 895.5710144042969), (36.68450927734375, 906.5302581787109)]


# Smooth the original track
smoothed_track = smooth_track(track, 2)

# Transform the track points
transformed_track = view_transformer.transform_points(np.array(track))

# Smooth the transformed track
smoothed_transformed_track = smooth_track(transformed_track, 2)

# Plot the smoothed and transformed tracks side-by-side
plot_smoothed_and_transformed_tracks(smoothed_track, smoothed_transformed_track)

# Plot the smoothed and transformed tracks side-by-side
plot_smoothed_and_transformed_tracks(smoothed_track, transformed_track)
