from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data=r"C:\Users\elbek\mol-bozor-demo-alif\Mol-bozor--person-3\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.6,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.9,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        cache=False,
        patience=30,
        close_mosaic=10,
        workers=4,
        device=0
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
