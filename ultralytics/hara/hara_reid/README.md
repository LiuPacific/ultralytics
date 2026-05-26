# Chicken OSNet ReID for DeepSORT

This project trains an OSNet ReID model for chicken crops and produces the cosine-distance cost matrix needed by DeepSORT.

The final DeepSORT-style result is:

```python
cost_matrix = cosine_distances(track_features, detection_features)
```

## 1. Install

```bash
cd chicken_osnet_deepsort_reid
pip install -r requirements.txt
```

If `pip install torchreid` fails, try:

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

## 2. Put raw images into `raw_images/`

Expected filename example:

```text
id_55_12_RGB_mock_frame_00_03_21.png
```

Meaning:

```text
id_55      -> chicken identity label
12         -> month
RGB        -> modality
mock       -> experiment group
00_03_21   -> frame time
```

## 3. Prepare data

```bash
python prepare_data.py --config config.yaml
```

This creates:

```text
prepared_reid/
    train/id_55/*.png
    val/id_55/*.png
    test/id_55/*.png
    metadata.csv
```

## 4. Train OSNet

```bash
python train.py --config config.yaml
```

Training outputs are saved in `outputs/`:

```text
best_osnet.pth                 # best checkpoint by validation loss
last_osnet.pth                 # latest checkpoint
training_log.csv               # epoch-by-epoch metrics
training_loss_curve.png         # train/validation loss curve
training_accuracy_curve.png     # classification accuracy and ReID rank-1 curve
validation_distance_curve.png   # same-ID vs different-ID distance curve
```

The training log contains:

```text
train_loss, train_ce_loss, train_triplet_loss, train_cls_acc
val_loss, val_ce_loss, val_triplet_loss, val_cls_acc
val_rank1
val_same_mean_dist, val_diff_mean_dist
suggested_threshold
```

Important interpretation:

- `val_cls_acc`: classifier-head validation accuracy.
- `val_rank1`: ReID retrieval accuracy. This is more relevant for DeepSORT.
- `val_same_mean_dist`: average cosine distance between images of the same chicken.
- `val_diff_mean_dist`: average cosine distance between images of different chickens.
- Good training should make `val_same_mean_dist` low and `val_diff_mean_dist` high.

## 5. Create full ReID report on test set

```bash
python report.py --config config.yaml
```

This creates:

```text
outputs/reid_report.csv
outputs/distance_distribution.png
```

Use this to choose a cosine-distance threshold for DeepSORT.

## 6. Calculate DeepSORT cost matrix

Example:

```bash
python extract_and_distance.py \
  --config config.yaml \
  --track_dir prepared_reid/test/id_55 \
  --detection_dir prepared_reid/test/id_56
```

This creates:

```text
outputs/cost_matrix.csv
outputs/track_features.npy
outputs/detection_features.npy
```

The key output is `cost_matrix.csv`. Small values mean similar appearance.

Example interpretation:

```text
0.08 -> very similar
0.45 -> probably different
0.80 -> very different
```

For your real chicken data, use `reid_report.csv` and `distance_distribution.png` to set the actual threshold.
