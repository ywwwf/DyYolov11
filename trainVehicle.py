from ultralytics import YOLO


def main():
    # Load a model
    model = YOLO(r"D:\Users\m\PycharmProjects\DyYolov11\yolo11n.yaml")

    # Train the model
    model.train(
        data="datasets_vehicle.yaml", epochs=1, imgsz=640, workers=0, batch=4, only_backbone=False, resume=False,
        pretrained=r"D:\Users\m\PycharmProjects\DyYolov11\runs\detect\train5\weights\last.pt"
                )


if __name__ == '__main__':
    main()
