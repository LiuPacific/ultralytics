yolo train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\HBB10min\hold1.yaml `
  imgsz=1280 `
  epochs=200 `
  patience=9999 `
  batch=11 `
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
  name=hbb_hold1_10min_test



Write-Host "Step 1: Starting hbb_hold5_5min"
yolo train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\HBB5min\hold5.yaml `
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
  name=hbb_hold5_5min

Write-Host "Step 2: hbb_hold5_5min finished"

yolo train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\HBB5min\hold6.yaml `
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
  name=hbb_hold6_5min



yolo obb train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l-obb.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\OBB5min\hold4.yaml `
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
  name=obb_hold4_5min


yolo obb train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l-obb.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\OBB5min\hold5.yaml `
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
  name=obb_hold5_5min





yolo obb train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l-obb.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\OBB5min\hold6.yaml `
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
  name=obb_hold6_5min



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



yolo obb train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l-obb.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\OBB10min\hold3.yaml `
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
  name=obb_hold3_10min



yolo obb train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l-obb.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\OBB10min\hold4.yaml `
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
  name=obb_hold4_10min



yolo obb train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l-obb.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\OBB10min\hold5.yaml `
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
  name=obb_hold5_10min




yolo obb train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l-obb.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\OBB10min\hold6.yaml `
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
  name=obb_hold6_10min

yolo train `
  model=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l.pt `
  data=C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\HBB5min\hold1.yaml `
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
  name=hbb_hold1_5min


  Write-Host "All commands finished"