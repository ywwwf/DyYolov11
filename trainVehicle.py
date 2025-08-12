from ultralytics import YOLO


def main():
    # 依赖yaml文件去创建模型，然后再加载参数
    model = YOLO(r"D:\Users\m\PycharmProjects\DyYolov11\yolo11n.yaml")

    # 训练main backbone
    # model.train(
    #     data="datasets_vehicle.yaml", epochs=1, imgsz=640, workers=0, batch=4, only_backbone=True, dynamicTrain=False,
    #     resume=False
    #             )

    # 训练Early exit
    # model.train(
    #     data="datasets_vehicle.yaml", epochs=1, imgsz=640, workers=0, batch=4, only_backbone=False,
    #     dynamicTrain=False, resume=False,
    #     pretrained=r"D:\Users\m\PycharmProjects\DyYolov11\runs\detect\train\weights\last.pt"
    #     )

    # 训练Router batch需要设置为1
    model.train(
        data="datasets_vehicle.yaml", epochs=1, imgsz=640, workers=0, batch=1, only_backbone=False, dynamicTrain=True,
        resume=False,
        pretrained=r"D:\Users\m\PycharmProjects\DyYolov11\runs\detect\train2\weights\last.pt"
    )

    # 根据验证集计算划分图片难度的阈值
    # model.get_thres(data="datasets_vehicle.yaml",
    #                 pretrained=r"D:\Users\m\PycharmProjects\DyYolov11\runs\detect\train7\weights\last.pt",
    #                 resume=False,
    #                 workers=0)

    # 测试需要在yaml文件中启用fast_inference_mode，同时设置所需的dy_thres


if __name__ == '__main__':
    main()
