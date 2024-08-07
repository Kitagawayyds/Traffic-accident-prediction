import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams

matplotlib.use('TkAgg')

# 设置 Matplotlib 使用黑体字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示问题


def show_frame_with_coordinates(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频文件")
        return

    # 将 BGR 转换为 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 初始化四个点的坐标
    points = [(50, 50), (150, 50), (150, 150), (50, 150)]

    # 创建图像窗口并显示第一帧
    fig, ax = plt.subplots()
    ax.imshow(frame_rgb)

    # 设置字体为黑体
    font_properties = {'family': 'sans-serif', 'weight': 'bold', 'size': 12}

    # 显示初始点
    scatter = ax.scatter(*zip(*points), color='blue', s=100, zorder=5)

    # 绘制半透明红色区域
    def draw_polygon():
        # 移除之前的多边形
        for patch in ax.patches:
            patch.remove()

        if len(points) == 4:
            polygon = patches.Polygon(points, closed=True, fill=True, color='red', alpha=0.5)
            ax.add_patch(polygon)

    draw_polygon()

    # 当前选择的点的索引
    current_point_index = None

    # 定义事件处理函数
    def on_click(event):
        nonlocal current_point_index

        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)

            # 如果有选中的点
            if current_point_index is not None:
                # 更新选定点的坐标
                points[current_point_index] = (x, y)
                scatter.set_offsets(points)
                draw_polygon()
                fig.canvas.draw_idle()
                current_point_index = None
            else:
                # 选择点
                distances = [((px - x) ** 2 + (py - y) ** 2) ** 0.5 for px, py in points]
                current_point_index = distances.index(min(distances))
                print(f"选择点: {current_point_index}, 点击位置: ({x}, {y})")
                ax.set_title(f"当前选择点: {current_point_index}", fontdict=font_properties)

    def on_motion(event):
        if current_point_index is not None and event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            # 更新选定点的坐标
            points[current_point_index] = (x, y)
            scatter.set_offsets(points)
            draw_polygon()
            fig.canvas.draw_idle()

    def on_release(event):
        nonlocal current_point_index
        if current_point_index is not None:
            print(f"点 {current_point_index} 更新为 ({int(event.xdata)}, {int(event.ydata)})")
            current_point_index = None

    # 连接事件
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # 显示图像
    plt.show()

    # 输出最终的四个点的坐标
    print("最终的四个点坐标:")
    for i, point in enumerate(points):
        print(f"点 {i}: {point}")

    # 释放视频文件
    cap.release()


# 示例用法
video_path = '../test2.mp4'
show_frame_with_coordinates(video_path)
