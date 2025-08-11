from ultralytics import YOLO


def main():
    # Load a model
    model = YOLO(r"D:\Users\m\PycharmProjects\DyYolov11\yolo11n.yaml")

    # Train the model
    # 训练路由器的时候 batch_size = 1
    model.train(
        data="datasets_vehicle.yaml", epochs=1, imgsz=640, workers=0, batch=1, only_backbone=False, dynamicTrain=True,
        resume=False,
        pretrained=r"D:\Users\m\PycharmProjects\DyYolov11\runs\detect\train6\weights\last.pt"
                )


if __name__ == '__main__':
    main()
