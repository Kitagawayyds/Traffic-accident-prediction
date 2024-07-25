from ultralytics import YOLO

def main():
    model = YOLO('yolov10x.pt')
    model.train(data="C:\\Users\kitag\PycharmProjects\AI\SCI\dataset\data.yaml",
                epochs=200, workers=8,
                batch=-1)
    model.val()
    model.export(format='onnx')

    model.predict()
if __name__ == "__main__":
    main()
