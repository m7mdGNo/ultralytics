# HRNet Pose (Heatmap + Offset)

This project now includes a custom HRNet-style keypoint model that:

- predicts a **class heatmap** and **x/y offset**
- uses a dataset folder with `train`, `val`, `test`
- reads `_annotations.csv` in each split with:
  `filename,width,height,class,xmin,ymin,xmax,ymax`

## Dataset Layout

```text
your_dataset/
  train/
    _annotations.csv
    image1.jpg
    image2.jpg
  val/
    _annotations.csv
    image3.jpg
  test/
    _annotations.csv
    image4.jpg
```

Each row in `_annotations.csv` is treated as one object.
The model learns keypoints at bbox centers (`(xmin+xmax)/2`, `(ymin+ymax)/2`).

## Python Usage

```python
from ultralytics import HRNetPose

# create/load model
model = HRNetPose("hrnet_pose.yaml")

# train
model.train(
    data="/absolute/path/to/your_dataset",
    imgsz=640,
    epochs=100,
    batch=16,
    workers=8,
    device=0,
)

# validate
metrics = model.val(data="/absolute/path/to/your_dataset")
print(metrics)

# predict
results = model.predict(
    source="/absolute/path/to/image_or_folder",
    conf=0.25,
    max_det=300,
)
```

## CLI Usage

You can run HRNet Pose directly with the Ultralytics CLI:

```bash
yolo train model=hrnet_pose.yaml data=/absolute/path/to/your_dataset imgsz=640 epochs=100 batch=16
```

```bash
yolo val model=hrnet_pose.yaml data=/absolute/path/to/your_dataset
```

```bash
yolo predict model=hrnet_pose.yaml source=/absolute/path/to/image_or_folder conf=0.25
```

You can also pass the explicit path:

```bash
yolo train model=ultralytics/cfg/models/v8/hrnet_pose.yaml data=/absolute/path/to/your_dataset
```

## What Is Returned During Predict

- predicted points come from heatmap peaks + offsets
- each point is returned as a keypoint in `Results.keypoints`
- small boxes are also attached to `Results.boxes` for easy visualization

## Notes

- task uses `pose` mode internally
- classes are inferred from `class` names found in split CSV files
- this implementation is optimized for center-keypoint style supervision from bbox CSV annotations
