# HRNet Pose (Heatmap + Offset)

This project now includes a custom HRNet-style keypoint model that:

- predicts a **class heatmap** and **x/y offset**
- uses a dataset folder with `train`, `valid` (or legacy `val`), `test`
- reads `_annotations.csv` in each split with:
  `filename,width,height,class,xmin,ymin,xmax,ymax`

## Dataset Layout

```text
your_dataset/
  train/
    _annotations.csv
    image1.jpg
    image2.jpg
  valid/
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

## Colab: avoid a stale `ultralytics` install

Colab often ships Ultralytics under `/usr/local/.../dist-packages/`. If you still see **`NameError: hm_p`** in `hrnet/nn.py`, you are **not** running your fork’s latest code.

**1) Reinstall from your fork (no cache), restart runtime:**

```bash
pip uninstall -y ultralytics
pip install --no-cache-dir "git+https://github.com/m7mdGNo/ultralytics.git"
```

**2) Verify the HRNet fix is present:**

```python
from ultralytics.models.hrnet import nn as hrnet_nn

assert getattr(hrnet_nn, "HRNET_NN_REV", 0) >= 2, "Old ultralytics; reinstall from GitHub"
print("HRNET_NN_REV", hrnet_nn.HRNET_NN_REV)
```

If the assertion fails, the notebook is still using an old wheel—repeat uninstall/install or pin a **commit SHA** in the `git+https://...` URL.

## Notes

- validation images live under **`valid/`** (recommended). If that folder has no `_annotations.csv`, the loader falls back to **`val/`**.
- task uses `pose` mode internally
- classes are inferred from `class` names found in split CSV files
- this implementation is optimized for center-keypoint style supervision from bbox CSV annotations
