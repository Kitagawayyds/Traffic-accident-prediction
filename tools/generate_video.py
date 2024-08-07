import cv2
import os


def create_video_from_images(image_folder, output_video, fps=25):
    # 获取文件夹中的所有图片文件
    images = [img for img in os.listdir(image_folder) if
              img.startswith("img") and img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # 确保图片按名称排序

    if not images:
        print("没有找到任何图片文件。")
        return

    # 获取图片的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        # 写入当前帧到视频文件
        video.write(frame)

    # 释放视频写入器
    video.release()
    print(f"视频已保存到 {output_video}")


# 使用示例
image_folder = 'C:\\Users\\26216\Downloads\DETRAC-test-data\Insight-MVT_Annotation_Test\MVI_40714'  # 替换为你的图片文件夹路径
output_video = 'test1.mp4'  # 替换为你的输出视频文件名
create_video_from_images(image_folder, output_video)
