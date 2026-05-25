# Chicken OSNet ReID for DeepSORT

End-to-end pipeline:

1. Prepare data from filenames such as `id_55_12_RGB_mock_frame_00_03_21.png`.
2. Train OSNet with cross-entropy + triplet loss.
3. Extract 512-D ReID embeddings.
4. Calculate cosine-distance cost matrix for DeepSORT.
5. Generate a report with distance distributions and Rank-1 retrieval.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

If `pip install torchreid` fails, use:

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

## Run

Put your images in `./raw_images`, then:

```bash
python prepare_data.py --config config.yaml
python train.py --config config.yaml
python extract_and_distance.py --config config.yaml --track_dir prepared_reid/test/id_55 --detection_dir prepared_reid/test/id_56
python report.py --config config.yaml
```

The DeepSORT cost matrix is produced in:

```text
outputs/cost_matrix.csv
```

The trained model is saved as:

```text
outputs/best_osnet.pth
```
