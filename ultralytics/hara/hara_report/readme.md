


- yaml
  - 5-min videos
    - hbb fold1
    - hbb fold2
    - hbb fold3
    - hbb fold4
    - hbb fold5
    - hbb fold6
    - obb fold1
    - obb fold2
    - obb fold3
    - obb fold4
    - obb fold5
    - obb fold6
  - 10-min videos
    - obb fold1
    - obb fold2
    - obb fold3
    - obb fold4
    - obb fold5
    - obb fold6
    - opt fold1
    - opt fold2
    - opt fold3
    - opt fold4
    - opt fold5
    - opt fold6


fix hypoparameters affected validation set.
```shell
yolo train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\HBB5min\hold4.yaml `
  imgsz=1280 `
  epochs=200 `
  patience=9999 `
  batch=10 `
  optimizer=SGD `
  lr0=0.01 `
  lrf=0.01 `
  cos_lr=False `
  warmup_epochs=3.0 `
  momentum=0.937 `
  weight_decay=0.0005 `
  hsv_h=0.015 `
  hsv_s=0.7 `
  hsv_v=0.4 `
  degrees=180 `
  translate=0.1 `
  scale=0.5 `
  shear=0.0 `
  perspective=0.0 `
  flipud=0.5 `
  fliplr=0.5 `
  mosaic=1.0 `
  mixup=0.0 `
  cutmix=0.0 `
  close_mosaic=10 `
  seed=0 `
  deterministic=True `
  name=hbb_hold4_5min
```



```sql
yolo obb train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l-obb.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\OBB5min\hold3.yaml `
  imgsz=1280 `
  epochs=200 `
  patience=9999 `
  batch=10 `
  optimizer=SGD `
  lr0=0.01 `
  lrf=0.01 `
  cos_lr=False `
  warmup_epochs=3.0 `
  momentum=0.937 `
  weight_decay=0.0005 `
  hsv_h=0.015 `
  hsv_s=0.7 `
  hsv_v=0.4 `
  degrees=180 `
  translate=0.1 `
  scale=0.5 `
  shear=0.0 `
  perspective=0.0 `
  flipud=0.5 `
  fliplr=0.5 `
  mosaic=1.0 `
  mixup=0.0 `
  cutmix=0.0 `
  close_mosaic=10 `
  seed=0 `
  deterministic=True `
  name=obb_hold3_5min
```


```shell
yolo obb train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l-obb.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\OBB10min\hold2.yaml `
  imgsz=1280 `
  epochs=200 `
  patience=9999 `
  batch=10 `
  optimizer=SGD `
  lr0=0.01 `
  lrf=0.01 `
  cos_lr=False `
  warmup_epochs=3.0 `
  momentum=0.937 `
  weight_decay=0.0005 `
  hsv_h=0.015 `
  hsv_s=0.7 `
  hsv_v=0.4 `
  degrees=180 `
  translate=0.1 `
  scale=0.5 `
  shear=0.0 `
  perspective=0.0 `
  flipud=0.5 `
  fliplr=0.5 `
  mosaic=1.0 `
  mixup=0.0 `
  cutmix=0.0 `
  close_mosaic=10 `
  seed=0 `
  deterministic=True `
  name=obb_hold2_10min
```