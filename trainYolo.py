from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov10n.pt')

    model.train(data='data/data.yaml', epochs=10, batch=2)

    model.val(data='data/data.yaml', batch=2)

    model.export(format='onnx')

