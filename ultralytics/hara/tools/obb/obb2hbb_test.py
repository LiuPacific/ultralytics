import json
import unittest
import tempfile
from pathlib import Path

from ultralytics.hara.tools.obb.obb2hbb import get_json_file_list,convert_json_file


class TestGetJsonFileList(unittest.TestCase):

    def test_get_json_file_list_non_recursive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_folder = Path(tmp_dir)

            json_1 = input_folder / "a.json"
            json_2 = input_folder / "b.json"
            txt_1 = input_folder / "c.txt"

            sub_folder = input_folder / "sub"
            sub_folder.mkdir()
            json_sub = sub_folder / "d.json"

            json_1.write_text("{}", encoding="utf-8")
            json_2.write_text("{}", encoding="utf-8")
            txt_1.write_text("not json", encoding="utf-8")
            json_sub.write_text("{}", encoding="utf-8")

            result = get_json_file_list(
                input_folder=input_folder,
                recursive=False
            )

            expected = sorted([json_1, json_2])

            self.assertEqual(result, expected)
            self.assertNotIn(txt_1, result)
            self.assertNotIn(json_sub, result)

    def test_get_json_file_list_recursive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_folder = Path(tmp_dir)

            json_1 = input_folder / "a.json"
            json_2 = input_folder / "b.json"

            sub_folder = input_folder / "sub"
            sub_folder.mkdir()
            json_sub = sub_folder / "d.json"

            txt_1 = input_folder / "c.txt"

            json_1.write_text("{}", encoding="utf-8")
            json_2.write_text("{}", encoding="utf-8")
            json_sub.write_text("{}", encoding="utf-8")
            txt_1.write_text("not json", encoding="utf-8")

            result = get_json_file_list(
                input_folder=input_folder,
                recursive=True
            )

            expected = sorted([json_1, json_2, json_sub])

            self.assertEqual(result, expected)
            self.assertNotIn(txt_1, result)


class TestConvertJsonFile(unittest.TestCase):

    def test_convert_json_file_obb_to_hbb(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            input_folder = root / "input"
            output_folder = root / "output"
            input_folder.mkdir()
            output_folder.mkdir()

            input_json_path = input_folder / "frame_00_00_01.json"

            test_data = {
                "version": "3.3.10",
                "flags": {},
                "shapes": [
                    {
                        "label": "1",
                        "score": None,
                        "points": [
                            [10.0, 20.0],
                            [30.0, 5.0],
                            [40.0, 25.0],
                            [15.0, 35.0],
                        ],
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": "rotation",
                        "direction": 1.234,
                        "flags": {},
                        "attributes": {},
                        "kie_linking": [],
                    },
                    {
                        "label": "3",
                        "score": None,
                        "points": [
                            [100.0, 200.0],
                            [150.0, 200.0],
                            [150.0, 260.0],
                            [100.0, 260.0],
                        ],
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": "rectangle",
                        "flags": {},
                        "attributes": {},
                        "kie_linking": [],
                    },
                ],
                "imagePath": "frame_00_00_01.png",
                "imageData": None,
                "imageHeight": 500,
                "imageWidth": 600,
                "description": "",
            }

            with open(input_json_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f, indent=2)

            output_json_path, converted_count, kept_count = convert_json_file(
                input_json_path=input_json_path,
                input_folder=input_folder,
                output_folder=output_folder
            )

            self.assertEqual(converted_count, 1)
            self.assertEqual(kept_count, 1)

            self.assertTrue(output_json_path.exists())
            self.assertEqual(
                output_json_path,
                output_folder / "frame_00_00_01.json"
            )

            with open(output_json_path, "r", encoding="utf-8") as f:
                converted_data = json.load(f)

            shapes = converted_data["shapes"]

            self.assertEqual(len(shapes), 2)

            converted_obb = shapes[0]
            kept_hbb = shapes[1]

            self.assertEqual(converted_obb["label"], "1")
            self.assertEqual(converted_obb["shape_type"], "rectangle")
            self.assertNotIn("direction", converted_obb)

            expected_hbb_points = [
                [10.0, 5.0],
                [40.0, 5.0],
                [40.0, 35.0],
                [10.0, 35.0],
            ]

            self.assertEqual(converted_obb["points"], expected_hbb_points)

            self.assertEqual(kept_hbb["label"], "3")
            self.assertEqual(kept_hbb["shape_type"], "rectangle")
            self.assertEqual(
                kept_hbb["points"],
                [
                    [100.0, 200.0],
                    [150.0, 200.0],
                    [150.0, 260.0],
                    [100.0, 260.0],
                ]
            )

    def test_convert_json_file_keep_relative_subfolder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            input_folder = root / "input"
            output_folder = root / "output"

            sub_folder = input_folder / "sub"
            sub_folder.mkdir(parents=True)
            output_folder.mkdir()

            input_json_path = sub_folder / "frame_sub.json"

            test_data = {
                "version": "3.3.10",
                "flags": {},
                "shapes": [
                    {
                        "label": "14",
                        "points": [
                            [20.0, 30.0],
                            [60.0, 10.0],
                            [80.0, 50.0],
                            [30.0, 70.0],
                        ],
                        "shape_type": "rotation",
                        "direction": 0.8,
                        "flags": {},
                        "attributes": {},
                        "kie_linking": [],
                    }
                ],
                "imagePath": "frame_sub.png",
                "imageData": None,
                "imageHeight": 200,
                "imageWidth": 300,
                "description": "",
            }

            with open(input_json_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f, indent=2)

            output_json_path, converted_count, kept_count = convert_json_file(
                input_json_path=input_json_path,
                input_folder=input_folder,
                output_folder=output_folder
            )

            expected_output_path = output_folder / "sub" / "frame_sub.json"

            self.assertEqual(output_json_path, expected_output_path)
            self.assertTrue(expected_output_path.exists())
            self.assertEqual(converted_count, 1)
            self.assertEqual(kept_count, 0)

            with open(expected_output_path, "r", encoding="utf-8") as f:
                converted_data = json.load(f)

            shape = converted_data["shapes"][0]

            self.assertEqual(shape["shape_type"], "rectangle")
            self.assertNotIn("direction", shape)

            expected_points = [
                [20.0, 10.0],
                [80.0, 10.0],
                [80.0, 70.0],
                [20.0, 70.0],
            ]

            self.assertEqual(shape["points"], expected_points)

    def test_convert_json_file_clamp_to_image_boundary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            input_folder = root / "input"
            output_folder = root / "output"
            input_folder.mkdir()
            output_folder.mkdir()

            input_json_path = input_folder / "out_of_boundary.json"

            test_data = {
                "version": "3.3.10",
                "flags": {},
                "shapes": [
                    {
                        "label": "1",
                        "points": [
                            [-10.0, 20.0],
                            [30.0, -5.0],
                            [120.0, 50.0],
                            [40.0, 150.0],
                        ],
                        "shape_type": "rotation",
                        "direction": 0.5,
                        "flags": {},
                        "attributes": {},
                        "kie_linking": [],
                    }
                ],
                "imagePath": "out_of_boundary.png",
                "imageData": None,
                "imageHeight": 100,
                "imageWidth": 100,
                "description": "",
            }

            with open(input_json_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f, indent=2)

            output_json_path, converted_count, kept_count = convert_json_file(
                input_json_path=input_json_path,
                input_folder=input_folder,
                output_folder=output_folder
            )

            self.assertEqual(converted_count, 1)
            self.assertEqual(kept_count, 0)

            with open(output_json_path, "r", encoding="utf-8") as f:
                converted_data = json.load(f)

            shape = converted_data["shapes"][0]

            expected_points = [
                [0, 0],
                [99, 0],
                [99, 99],
                [0, 99],
            ]

            self.assertEqual(shape["points"], expected_points)


if __name__ == "__main__":
    unittest.main()