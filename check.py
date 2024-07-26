from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:\\Users\kitag\PycharmProjects\AI\SCI/runs\detect/train\weights/best.pt')

    path = 'test.mp4'

    results = model.predict(source=path, save=True, show=True)
