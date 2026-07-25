import argparse
import json
import math
from copy import deepcopy
from pathlib import Path


EXPECTED_CHICKEN_COUNT = 15


def get_json_file_list(input_folder, recursive=False):
    input_folder = Path(input_folder)

    if recursive:
        return sorted(input_folder.rglob("*.json"))

    return sorted(input_folder.glob("*.json"))


def normalize_pair_key(path):
    path = Path(path)
    stem = path.stem

    if stem.endswith("_hbb"):
        stem = stem[:-4]
    elif stem.endswith("_obb"):
        stem = stem[:-4]

    return path.parent / stem


def get_bbox_center(shape):
    points = shape.get("points", [])
    if len(points) != 4:
        raise ValueError(f"Shape {shape.get('label')} does not have 4 points.")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    return (
        (min(xs) + max(xs)) / 2.0,
        (min(ys) + max(ys)) / 2.0,
    )


def build_distance_table(hbb_shapes, obb_shapes):
    distance_table = []

    for hbb_index, hbb_shape in enumerate(hbb_shapes):
        hbb_center = get_bbox_center(hbb_shape)

        for obb_index, obb_shape in enumerate(obb_shapes):
            obb_center = get_bbox_center(obb_shape)
            distance = math.dist(hbb_center, obb_center)
            distance_table.append((distance, hbb_index, obb_index))

    distance_table.sort(key=lambda item: item[0])
    return distance_table


def match_shapes_by_center(hbb_shapes, obb_shapes):
    matched_hbb_indices = set()
    matched_obb_indices = set()
    matches = []

    for distance, hbb_index, obb_index in build_distance_table(hbb_shapes, obb_shapes):
        if hbb_index in matched_hbb_indices or obb_index in matched_obb_indices:
            continue

        matched_hbb_indices.add(hbb_index)
        matched_obb_indices.add(obb_index)
        matches.append((hbb_index, obb_index, distance))

        if len(matched_hbb_indices) == len(hbb_shapes) or len(matched_obb_indices) == len(obb_shapes):
            break

    unmatched_hbb_indices = sorted(set(range(len(hbb_shapes))) - matched_hbb_indices)
    unmatched_obb_indices = sorted(set(range(len(obb_shapes))) - matched_obb_indices)

    return matches, unmatched_hbb_indices, unmatched_obb_indices


def label_hbb_by_obb(hbb_json_path, obb_json_path, output_json_path=None, expected_chicken_count=EXPECTED_CHICKEN_COUNT):
    hbb_json_path = Path(hbb_json_path)
    obb_json_path = Path(obb_json_path)
    output_json_path = Path(output_json_path) if output_json_path else hbb_json_path

    with open(hbb_json_path, "r", encoding="utf-8") as hbb_file:
        hbb_data = json.load(hbb_file)

    with open(obb_json_path, "r", encoding="utf-8") as obb_file:
        obb_data = json.load(obb_file)

    hbb_shapes = hbb_data.get("shapes", [])
    obb_shapes = obb_data.get("shapes", [])
    warnings = []

    if len(hbb_shapes) != expected_chicken_count:
        warnings.append(
            f"[WARNING] HBB chicken count is {len(hbb_shapes)}, expected {expected_chicken_count}: {hbb_json_path.name}"
        )

    if len(obb_shapes) != expected_chicken_count:
        warnings.append(
            f"[WARNING] OBB chicken count is {len(obb_shapes)}, expected {expected_chicken_count}: {obb_json_path.name}"
        )

    matches, unmatched_hbb_indices, unmatched_obb_indices = match_shapes_by_center(hbb_shapes, obb_shapes)

    new_hbb_data = deepcopy(hbb_data)
    new_hbb_shapes = deepcopy(hbb_shapes)

    for hbb_index, obb_index, _distance in matches:
        new_hbb_shapes[hbb_index]["label"] = str(obb_shapes[obb_index].get("label"))

    for hbb_index in unmatched_hbb_indices:
        warnings.append(
            f"[WARNING] HBB shape index {hbb_index} ({hbb_shapes[hbb_index].get('label')}) was not matched to any OBB shape."
        )

    for obb_index in unmatched_obb_indices:
        warnings.append(
            f"[WARNING] OBB shape index {obb_index} ({obb_shapes[obb_index].get('label')}) was not matched to any HBB shape."
        )

    new_hbb_data["shapes"] = new_hbb_shapes
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as output_file:
        json.dump(new_hbb_data, output_file, indent=2, ensure_ascii=False)

    return output_json_path, matches, warnings


def build_json_index(json_paths, root_folder):
    root_folder = Path(root_folder)
    index = {}

    for json_path in json_paths:
        relative_path = Path(json_path).relative_to(root_folder)
        pair_key = normalize_pair_key(relative_path)
        index[pair_key] = Path(json_path)

    return index


def label_hbb_folder_by_obb(
    hbb_folder,
    obb_folder,
    output_folder=None,
    recursive=False,
    expected_chicken_count=EXPECTED_CHICKEN_COUNT,
):
    hbb_folder = Path(hbb_folder)
    obb_folder = Path(obb_folder)
    output_folder = Path(output_folder) if output_folder else hbb_folder

    hbb_json_paths = get_json_file_list(hbb_folder, recursive=recursive)
    obb_json_paths = get_json_file_list(obb_folder, recursive=recursive)

    hbb_index = build_json_index(hbb_json_paths, hbb_folder)
    obb_index = build_json_index(obb_json_paths, obb_folder)

    pair_keys = sorted(set(hbb_index) | set(obb_index))
    results = []
    warnings = []

    for pair_key in pair_keys:
        hbb_json_path = hbb_index.get(pair_key)
        obb_json_path = obb_index.get(pair_key)

        if hbb_json_path is None:
            warnings.append(f"[WARNING] Missing HBB JSON for pair key: {pair_key.as_posix()}")
            continue

        if obb_json_path is None:
            warnings.append(f"[WARNING] Missing OBB JSON for pair key: {pair_key.as_posix()}")
            continue

        relative_hbb_path = hbb_json_path.relative_to(hbb_folder)
        output_json_path = output_folder / relative_hbb_path

        labeled_output_json_path, matches, file_warnings = label_hbb_by_obb(
            hbb_json_path=hbb_json_path,
            obb_json_path=obb_json_path,
            output_json_path=output_json_path,
            expected_chicken_count=expected_chicken_count,
        )

        results.append(
            {
                "hbb_json_path": hbb_json_path,
                "obb_json_path": obb_json_path,
                "output_json_path": labeled_output_json_path,
                "match_count": len(matches),
                "warnings": file_warnings,
            }
        )
        warnings.extend(file_warnings)

    return results, warnings


def main():
    # output_json_path, matches, warnings = label_hbb_by_obb(
    #     hbb_json_path=args.hbb_json,
    #     obb_json_path=args.obb_json,
    #     output_json_path=args.output_json,
    #     expected_chicken_count=args.expected_chicken_count,
    # )
    #
    # print(f"[OK] Matched {len(matches)} HBB shapes using OBB IDs.")
    # print(f"[OK] Output JSON: {output_json_path}")
    #
    # for warning in warnings:
    #     print(warning)
    # return

    # if folder_mode:
    results, warnings = label_hbb_folder_by_obb(
        hbb_folder=r"D:\chicken_project\experiment5reid\reid_hbb_training\20251217T000000Z_20251217T003000Z_RGB_mock 46-60",
        obb_folder=r"D:\chicken_project\experiment5reid\reid_obb_training\20251217T000000Z_20251217T003000Z_RGB_mock 46-60",
        output_folder=r"D:\chicken_project\experiment5reid\reid_hbb_training\20251217T000000Z_20251217T003000Z_RGB_mock 46-60",
        recursive=False,
        expected_chicken_count=15,
    )

    print(f"[OK] Processed {len(results)} HBB JSON files.")

    for result in results:
        print(
            f"[OK] {result['hbb_json_path'].name} <- {result['obb_json_path'].name} | "
            f"matched: {result['match_count']} | output: {result['output_json_path']}"
        )

    for warning in warnings:
        print(warning)
    return



if __name__ == "__main__":
    main()
