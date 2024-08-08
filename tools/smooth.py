import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

matplotlib.use('TkAgg')

def smooth_track(track, sigma):
    if len(track) < 3:
        return track

    x_coords = [p[0] for p in track]
    y_coords = [p[1] for p in track]

    x_smooth = gaussian_filter1d(x_coords, sigma)
    y_smooth = gaussian_filter1d(y_coords, sigma)

    smoothed_track = list(zip(x_smooth, y_smooth))

    return smoothed_track

def plot_tracks(original_track, smoothed_track):
    original_x = [p[0] for p in original_track]
    original_y = [p[1] for p in original_track]

    smoothed_x = [p[0] for p in smoothed_track]
    smoothed_y = [p[1] for p in smoothed_track]

    plt.figure(figsize=(10, 6))

    # Plot original track
    plt.plot(original_x, original_y, 'o-', label='Original Track', color='blue')

    # Plot smoothed track
    plt.plot(smoothed_x, smoothed_y, 'x-', label='Smoothed Track', color='red')

    plt.xlabel('X Coordinate (pixels)')
    plt.ylabel('Y Coordinate (pixels)')
    plt.title('Comparison of Original and Smoothed Track')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
track = [(169.26242065429688, 633.3276519775391), (169.27770598977804, 633.3187255859375), (168.01373781333677, 633.2021942138672), (167.65962576307356, 634.9336090087891), (166.596793115139, 636.6077117919922), (165.76338504999876, 636.762451171875), (165.09966130182147, 638.4427947998047), (164.6685304120183, 640.0843200683594), (164.40738053619862, 640.2413177490234), (164.18057610094547, 641.7663269042969), (163.0901551991701, 641.7286987304688), (162.53284326195717, 641.6690216064453), (162.192876085639, 642.1328277587891), (160.91139908134937, 642.3611755371094), (160.16455802321434, 642.3351898193359), (159.7072895243764, 642.5381317138672), (159.19057470560074, 641.2679901123047), (158.917214512825, 640.3932952880859), (158.43085968494415, 641.1295928955078), (158.14915096759796, 640.1119842529297)]
sigma = 1.0

smoothed_track = smooth_track(track, sigma)
plot_tracks(track, smoothed_track)
