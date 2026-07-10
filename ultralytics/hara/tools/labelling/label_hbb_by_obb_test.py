import json
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("label_hbb_by_obb.py")
MODULE_SPEC = importlib.util.spec_from_file_location("label_hbb_by_obb", MODULE_PATH)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)
label_hbb_by_obb = MODULE.label_hbb_by_obb
label_hbb_folder_by_obb = MODULE.label_hbb_folder_by_obb


class TestLabelHbbByObb(unittest.TestCase):

    def test_label_hbb_by_obb_on_fixture_files(self):
        fixture_folder = Path(__file__).parent / ".test"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            hbb_input = root / "frame_00_12_11_hbb.json"
            obb_input = root / "frame_00_12_11_obb.json"
            output_path = root / "frame_00_12_11_hbb_labeled.json"

            hbb_input.write_text(
                (fixture_folder / "frame_00_12_11_hbb.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            obb_input.write_text(
                (fixture_folder / "frame_00_12_11_obb.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            output_json_path, matches, warnings = label_hbb_by_obb(
                hbb_json_path=hbb_input,
                obb_json_path=obb_input,
                output_json_path=output_path,
            )

            self.assertEqual(output_json_path, output_path)
            self.assertEqual(len(matches), 15)
            self.assertEqual(warnings, [])

            with open(output_json_path, "r", encoding="utf-8") as output_file:
                labeled_data = json.load(output_file)

            labels = [shape["label"] for shape in labeled_data["shapes"]]

            self.assertEqual(
                labels,
                ["7", "3", "13", "8", "15", "4", "2", "12", "1", "5", "11", "9", "10", "6", "14"],
            )
            self.assertIn("15", labels)

    def test_label_hbb_by_obb_warns_for_count_mismatch_and_unmatched_shapes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            hbb_input = root / "hbb.json"
            obb_input = root / "obb.json"

            hbb_data = {
                "shapes": [
                    {
                        "label": "chicken",
                        "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                        "shape_type": "rectangle",
                    },
                    {
                        "label": "chicken",
                        "points": [[20.0, 0.0], [30.0, 0.0], [30.0, 10.0], [20.0, 10.0]],
                        "shape_type": "rectangle",
                    },
                ]
            }
            obb_data = {
                "shapes": [
                    {
                        "label": "15",
                        "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                        "shape_type": "rotation",
                    }
                ]
            }

            hbb_input.write_text(json.dumps(hbb_data), encoding="utf-8")
            obb_input.write_text(json.dumps(obb_data), encoding="utf-8")

            output_json_path, matches, warnings = label_hbb_by_obb(
                hbb_json_path=hbb_input,
                obb_json_path=obb_input,
            )

            self.assertEqual(output_json_path, hbb_input)
            self.assertEqual(len(matches), 1)
            self.assertEqual(len(warnings), 3)
            self.assertIn("HBB chicken count is 2, expected 15", warnings[0])
            self.assertIn("OBB chicken count is 1, expected 15", warnings[1])
            self.assertIn("HBB shape index 1", warnings[2])

            with open(hbb_input, "r", encoding="utf-8") as output_file:
                labeled_data = json.load(output_file)

            self.assertEqual(labeled_data["shapes"][0]["label"], "15")
            self.assertEqual(labeled_data["shapes"][1]["label"], "chicken")

    def test_label_hbb_folder_by_obb_matches_by_normalized_file_name(self):
        fixture_folder = Path(__file__).parent / ".test"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            hbb_folder = root / "hbb"
            obb_folder = root / "obb"
            output_folder = root / "output"
            hbb_folder.mkdir()
            obb_folder.mkdir()

            (hbb_folder / "frame_00_12_11_hbb.json").write_text(
                (fixture_folder / "frame_00_12_11_hbb.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (obb_folder / "frame_00_12_11_obb.json").write_text(
                (fixture_folder / "frame_00_12_11_obb.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            results, warnings = label_hbb_folder_by_obb(
                hbb_folder=hbb_folder,
                obb_folder=obb_folder,
                output_folder=output_folder,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(warnings, [])
            self.assertEqual(results[0]["match_count"], 15)
            self.assertEqual(results[0]["output_json_path"], output_folder / "frame_00_12_11_hbb.json")

            labeled_data = json.loads(results[0]["output_json_path"].read_text(encoding="utf-8"))
            labels = [shape["label"] for shape in labeled_data["shapes"]]
            self.assertIn("15", labels)

    def test_label_hbb_folder_by_obb_warns_for_missing_pairs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            hbb_folder = root / "hbb"
            obb_folder = root / "obb"
            hbb_folder.mkdir()
            obb_folder.mkdir()

            hbb_data = {
                "shapes": [
                    {
                        "label": "chicken",
                        "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                        "shape_type": "rectangle",
                    }
                ]
            }
            obb_data = {
                "shapes": [
                    {
                        "label": "15",
                        "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                        "shape_type": "rotation",
                    }
                ]
            }

            (hbb_folder / "frame_a_hbb.json").write_text(json.dumps(hbb_data), encoding="utf-8")
            (obb_folder / "frame_b_obb.json").write_text(json.dumps(obb_data), encoding="utf-8")

            results, warnings = label_hbb_folder_by_obb(
                hbb_folder=hbb_folder,
                obb_folder=obb_folder,
            )

            self.assertEqual(results, [])
            self.assertEqual(len(warnings), 2)
            self.assertIn("Missing OBB JSON for pair key: frame_a", warnings[0])
            self.assertIn("Missing HBB JSON for pair key: frame_b", warnings[1])


if __name__ == "__main__":
    unittest.main()
