"""Focused contracts for deterministic Nymeria inference prompts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from native_phase_training.nymeria_i2v_prompt import (
    DESCRIPTION_PLACEHOLDER,
    build_nymeria_i2v_negative_prompt,
    build_nymeria_i2v_prompt,
    compact_prompt,
    load_template,
    normalize_native_structured_prompt,
    write_prompt_artifacts,
)
from native_phase_training.sanitize_prefix_inference_inputs import sanitize_record


class NymeriaI2VPromptTest(unittest.TestCase):
    def test_only_action_description_depends_on_caption(self) -> None:
        description = "The person walks around the table and raises her right hand."
        template = load_template()
        prompt = build_nymeria_i2v_prompt(
            description,
            num_frames=97,
            fps=20,
            height=256,
            width=256,
        )

        self.assertIn(DESCRIPTION_PLACEHOLDER, template["actions"][0]["description"])
        self.assertEqual(
            prompt["actions"][0]["description"],
            description,
        )
        self.assertEqual(compact_prompt(prompt).count(description), 1)
        self.assertEqual(prompt["actions"][0]["time"], "0:00-0:04")
        self.assertEqual(prompt["duration"], "4s")
        self.assertEqual(prompt["fps"], 20.0)
        self.assertEqual(prompt["resolution"], {"H": 256, "W": 256})
        self.assertEqual(prompt["aspect_ratio"], "1,1")
        self.assertEqual(
            prompt["cinematography"]["camera_angle"],
            "Natural eye-level viewpoint of the unseen camera wearer",
        )
        self.assertEqual(
            prompt["cinematography"]["lens_focal_length"],
            "Wide-angle fisheye lens consistent with the conditioning frame",
        )

    def test_negative_template_is_nymeria_specific_and_uses_request_metadata(self) -> None:
        prompt = build_nymeria_i2v_negative_prompt(
            num_frames=97,
            fps=20,
            height=256,
            width=256,
        )
        encoded = compact_prompt(prompt).lower()
        self.assertIn("external camera", encoded)
        self.assertNotIn("contradicts", encoded)
        self.assertNotIn("locked static", encoded)
        self.assertIn("headset", encoded)
        self.assertNotIn("subjects", prompt)
        self.assertNotIn("number_of_subjects", encoded)
        self.assertNotIn("driving", encoded)
        self.assertNotIn("car", encoded)
        self.assertNotIn("actions", prompt)
        self.assertEqual(prompt["fps"], 20.0)
        self.assertEqual(prompt["resolution"], {"H": 256, "W": 256})

    def test_empty_description_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "description must be non-empty"):
            build_nymeria_i2v_prompt("  ", num_frames=97, fps=20, height=256, width=256)

    def test_edge_sanitizer_templates_plain_i2v_but_preserves_structured_json(self) -> None:
        base = {
            "model_mode": "image2video",
            "name": "sample_image2video",
            "num_frames": 97,
            "resolution": "256",
            "aspect_ratio": "1,1",
            "fps": 20,
            "shift": 10.0,
            "num_steps": 35,
            "guidance": 6.0,
            "vision_path": "/tmp/frame.png",
        }
        templated = sanitize_record(
            {**base, "prompt": "C walks around the table."},
            "image2video",
            model_family="edge",
            replace_standalone_c=True,
        )
        prompt = json.loads(templated["prompt"])
        self.assertEqual(
            prompt["actions"][0]["description"],
            "The camera wearer walks around the table.",
        )
        negative = json.loads(templated["negative_prompt"])
        self.assertIn("external camera", compact_prompt(negative).lower())
        self.assertEqual(templated["negative_metadata_mode"], "none")
        self.assertFalse(templated["negative_prompt_keep_metadata"])

        structured = compact_prompt(
            build_nymeria_i2v_prompt(
                "The camera wearer sits down.",
                num_frames=97,
                fps=20,
                height=256,
                width=256,
            )
        )
        preserved = sanitize_record(
            {**base, "prompt": structured},
            "image2video",
            model_family="edge",
            replace_standalone_c=True,
        )
        self.assertEqual(preserved["prompt"], structured)

        lowercase = sanitize_record(
            {**base, "prompt": "c turns left."},
            "image2video",
            model_family="edge",
            replace_standalone_c=True,
        )
        self.assertEqual(
            json.loads(lowercase["prompt"])["actions"][0]["description"],
            "The camera wearer turns left.",
        )

    def test_native_structured_prompt_normalization_matches_runtime_format(self) -> None:
        source = '{"actions":[{"description":"move"}],"fps":99,"resolution":{"H":1,"W":2}}'
        normalized = normalize_native_structured_prompt(
            source,
            num_frames=97,
            fps=20,
            height=256,
            width=256,
            aspect_ratio="1,1",
        )
        self.assertEqual(
            normalized,
            json.dumps(
                {
                    "actions": [{"description": "move"}],
                    "fps": 20.0,
                    "resolution": {"H": 256, "W": 256},
                    "duration": "4s",
                    "aspect_ratio": "1,1",
                }
            ),
        )

    def test_prompt_artifacts_are_saved_beside_output(self) -> None:
        positive = compact_prompt(
            build_nymeria_i2v_prompt(
                "The camera wearer walks forward.",
                num_frames=97,
                fps=20,
                height=256,
                width=256,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_prompt_artifacts(
                root,
                positive_prompt=positive,
                negative_prompt='{"avoid":"flicker"}',
            )
            self.assertTrue((root / "positive_prompt.json").is_file())
            self.assertTrue((root / "negative_prompt.json").is_file())
            self.assertTrue((root / "prompt_manifest.json").is_file())
            self.assertEqual(manifest["positive"]["format"], "structured_json")


if __name__ == "__main__":
    unittest.main()
