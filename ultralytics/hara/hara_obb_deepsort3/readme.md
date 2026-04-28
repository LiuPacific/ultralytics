# introduction

1. label data using X-any-labeling
    1. label every chicken as 0
2. export YOLO-pose format.
   1. add a file `predefined_class.txt`
   2. add `0` into the file to indicate which label will be exported.
3. divide training files into `train`, `val`, and `test` directories. 

# training


yolo train data=C:/Users/tliu25/workspace/ultralytics/ultralytics/hara/hara_obb_deepsort3/chicken.yaml model=C:/Users/tliu25/workspace/ultralytics/ultralytics/hara/weights/yolov8m-obb-chicken-0401.pt epochs=200 imgsz=1280 batch=16 workers=4 name=yolov8m-obb-chicken-0426

# prediction

- 1 image
yolo predict model=C:/Users/tliu25/workspace/ultralytics/ultralytics/hara/weights/yolov8m-obb-chicken.pt data=C:/Users/tliu25/workspace/ultralytics/ultralytics/hara/hara_obb/chicken.yaml source=D:/chicken_project/experiment3obb/prediction_20250825T104000Z_20250825T110000Z_RGB_mock/frame_00_20_01.png conf=0.5 line_width=1
  

- video

```shell
yolo predict model=C:/Users/tliu25/workspace/ultralytics/ultralytics/hara/weights/yolov8m-obb-chicken.pt data=C:/Users/tliu25/workspace/ultralytics/ultralytics/hara/hara_obb/chicken.yaml source=D:/chicken_project/experiment3obb/prediction/sick_20250825T232000Z_20250825T234000Z_training.mkv conf=0.5 line_width=3 show
yolo predict model=C:/Users/tliu25/workspace/ultralytics/ultralytics/hara/weights/yolov8m-obb-chicken.pt data=C:/Users/tliu25/workspace/ultralytics/ultralytics/hara/hara_obb/chicken.yaml source=D:/chicken_project/experiment3obb/prediction/mock_20250825T232000Z_20250825T234000Z_prediction.mkv conf=0.5 line_width=3 show
```

