from pathlib import Path
import shutil

folder = Path(r"D:\chicken_project\experiment5reid\reid_training_hbb\20251007T000000Z_20251007T010000Z_RGB_mock 76-90")  # change this
backup_folder = Path(r"D:\chicken_project\experiment5reid\reid_training_hbb\20251007T000000Z_20251007T010000Z_RGB_mock 76-90\keep")
backup_folder.mkdir(exist_ok=True)

frame_names = sorted({
    p.stem for p in folder.iterdir()
    if p.suffix.lower() in [".png", ".json"]
})

for i, frame_name in enumerate(frame_names):
    if i % 2 == 0:
        for ext in [".png", ".json"]:
            file_path = folder / f"{frame_name}{ext}"
            if file_path.exists():
                print("Moving:", file_path.name)
                shutil.move(str(file_path), str(backup_folder / file_path.name))

print("Done.")