"""Export and verify the exact captions used by the selected native evaluation.

Standard library only. No model inference, sampling, GT captions, video, or
credentials are exported. The accepted prediction receipts bind every caption
to the actual selected experiment, rather than just a matching prompt name.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

VARIANT = "Q40_qwen38_same_prompt_full32"
SELECTED = "G08_Q40_v1p5_t0p5"
CASES_SHA256 = "70a586033687177cad2a5b5cc23b6c3b3dea35131b430ad8f2d0cc7ebdb2a667"
SELECTION_SHA256 = "47288f230f6b21835f30d5874641b24b0129ab669f99e2f0d03cb1dcf5ced2df"
PROMPT_SHA256 = "392d3e3082ccb7805280b959703271fc1089602c27853df6cdf0b65f3554d688"
CHECKPOINT_SHA256 = "78dd00a6df0d95a96dfefcd0a68ad376e6dff8968bb48f2b13d4be18551ffddc"
MODEL_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
RECEIPTS_SHA256 = "7b2e6d6f2420f0716c9da8f75e7f44156c01e74e17c247b9b28d4457715b7a7d"
ID_PATTERN = re.compile(r"nymeria_[0-9a-f]{24}\Z")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path):
    return json.loads(path.read_bytes())


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def validate_records(records):
    require(len(records) == 1420, "Expected all1420 random20 windows, not the first71 subset")
    require(len({r["id"] for r in records}) == 1420, "Duplicate window IDs")
    recordings = {}
    for row in records:
        require(bool(ID_PATTERN.fullmatch(row["id"])), "Invalid window ID")
        recordings[row["uuid"]] = recordings.get(row["uuid"], 0) + 1
        require(isinstance(row["caption"], str) and bool(row["caption"].strip()), "Empty caption")
        require(digest(row["caption"].encode("utf-8")) == row["caption_utf8_sha256"], "Caption text hash mismatch")
        require(row["frames"] == 97 and row["fps"] == 20, "Unexpected temporal protocol")
        require(row["end_exclusive"] - row["start"] == 97, "Unexpected window bounds")
        require(isinstance(row["seed"], int), "Missing native per-case seed")
    require(len(recordings) == 71 and set(recordings.values()) == {20}, "Expected20 windows for each of71 recordings")


def verify_bundle(directory: Path) -> dict:
    expected = read(directory / "SHA256SUMS.json")
    actual_files = {p.name for p in directory.iterdir() if p.is_file()} - {"SHA256SUMS.json"}
    require(set(expected) == actual_files, "Bundle file set differs from its checksum manifest")
    for name, sha in expected.items():
        require(Path(name).name == name, "Unsafe checksum path")
        require(digest((directory / name).read_bytes()) == sha, f"Checksum mismatch: {name}")
    data = read(directory / "captions_random20.json")
    require(data["cases_sha256"] == CASES_SHA256 and data["selection_sha256"] == SELECTION_SHA256, "Wrong cohort")
    require(data["caption_variant"] == VARIANT and data["selected_condition"] == SELECTED, "Wrong caption experiment")
    records = data["records"]
    validate_records(records)
    require(digest(("\n".join(r["id"] for r in records) + "\n").encode()) == data["ordered_ids_sha256"], "Case order changed")
    require(read(directory / "captions_by_id.json") == {r["id"]: r["caption"] for r in records}, "Caption lookup differs")
    with (directory / "captions_random20.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1420, "Incomplete CSV")
    for record, row in zip(records, rows, strict=True):
        require(row == {key: str(value) for key, value in record.items()}, "CSV caption or case metadata differs")
    require(digest((directory / "Q40_prompt.txt").read_bytes()) == PROMPT_SHA256, "Prompt template changed")
    recipe = read(directory / "recipe.json")
    require(recipe["video_CFG"] == 1.5 and recipe["text_CFG"] == 0.5, "Wrong selected guidance")
    require(recipe["checkpoint_sha256"] == CHECKPOINT_SHA256, "Wrong model checkpoint")
    return {"passed": True, "windows": 1420, "recordings": 71, "captions_equal_in_JSON_and_CSV": True}


def export_bundle(recipe_path: Path, directory: Path, *, cases_path: Path | None = None,
                  captions_root: Path | None = None, receipts_path: Path | None = None) -> dict:
    require(not directory.exists(), "Refusing to overwrite an existing handoff")
    archive = directory.with_name(directory.name + ".zip")
    require(not archive.exists(), "Refusing to overwrite an existing archive")
    recipe_source = read(recipe_path)
    require(recipe_source["passed"] and recipe_source["video_CFG"] == 1.5 and recipe_source["text_CFG"] == 0.5, "Recipe is not the accepted winner")
    require(recipe_source["motion_checkpoint"]["SHA256"] == CHECKPOINT_SHA256, "Wrong EMA checkpoint")
    cases_path = cases_path or Path(recipe_source["manifest"]["path"])
    captions_root = captions_root or Path(recipe_source["unchanged_actual1420_captions"])
    receipts_path = receipts_path or Path(recipe_source["prediction_root"]).parents[2] / "report/PREDICTION_AND_CAPTION_RECEIPTS.json"
    case_bytes = cases_path.read_bytes()
    require(digest(case_bytes) == CASES_SHA256, "Cases are not the exact frozen random20 manifest")
    prompt_path = Path(recipe_source["prompt_source"]["path"])
    prompt = prompt_path.read_bytes()
    require(digest(prompt) == PROMPT_SHA256, "Captioning prompt differs from the evaluated prompt")
    require(digest(Path(recipe_source["native_score_report"]["path"]).read_bytes()) == recipe_source["native_score_report"]["SHA256"], "Accepted result report changed")
    require(digest(receipts_path.read_bytes()) == RECEIPTS_SHA256, "Wrong accepted prediction receipts")
    receipts = {r["id"]: r for r in read(receipts_path) if r["variant"] == SELECTED}
    cases = json.loads(case_bytes)
    require(len(receipts) == 1420 and set(receipts) == {c["id"] for c in cases}, "Prediction receipts do not cover the exact cohort")
    records = []
    for index, case in enumerate(cases):
        caption_path = captions_root / str(index % 4) / (case["id"] + ".json")
        raw = caption_path.read_bytes()
        caption = json.loads(raw)
        meta = caption["_meta"]
        require(digest(raw) == receipts[case["id"]]["caption_SHA256"], "Caption differs from what the winning model actually consumed")
        require(caption["id"] == case["id"] and meta["variant"] == VARIANT, "Wrong caption identity")
        require(meta["prompt_sha256"] == PROMPT_SHA256 and meta["model_revision"] == MODEL_REVISION, "Wrong prompt or VLM revision")
        require(caption["GT_caption_provided"] is False and caption["GT_motion_provided"] is False, "GT-reference data is not a generated caption")
        require(meta["target_GT_caption_provided"] is False and meta["GT_camera_sent_to_VLM"] is False, "Invalid VLM input provenance")
        require(meta["RGB_provided"] is True and meta["case_identifier_sent_to_VLM"] is False, "Invalid caption input provenance")
        row = {key: case[key] for key in ["id", "uuid", "start", "end_exclusive", "frames", "fps", "seed"]}
        row.update(caption=caption["caption"], caption_seed=meta["case_seed"],
                   caption_utf8_sha256=digest(caption["caption"].encode("utf-8")), source_caption_sha256=digest(raw))
        records.append(row)
    validate_records(records)
    data = dict(schema="nymeria.random20.generated_captions.v1", caption_variant=VARIANT,
                selected_condition=SELECTED, cases_sha256=CASES_SHA256, selection_sha256=SELECTION_SHA256,
                ordered_ids_sha256=digest(("\n".join(r["id"] for r in records) + "\n").encode()), records=records)
    portable_recipe = dict(caption_model="Qwen3.8-27B", model_revision=MODEL_REVISION,
                          frames=32, pixels=512, temperature=1.0, top_p=0.95, top_k=20,
                          repetition_penalty=1.0, canonical_caption_GPU="A10080GB", caption_batch=4,
                          prompt_sha256=PROMPT_SHA256, video_CFG=1.5, text_CFG=0.5,
                          checkpoint_sha256=CHECKPOINT_SHA256,
                          original_full_checkpoint_sha256=recipe_source["original_full_checkpoint_SHA256"],
                          weights="EMA", checkpoint_step=200000, sampler="native UniPC", steps=35,
                          motion_video_resolution=256, supplied_shape=True, camera_condition=False,
                          text_routing="motion expert only; generator caption remains empty",
                          no_per_case_caption_or_CFG_selection=True,
                          generated_metrics=recipe_source["all10_and_extra_native_metrics"],
                          actual_GT_metrics=recipe_source["actual_GT_metrics"],
                          matched_video_metrics=recipe_source["matched_native_video"],
                          matched_GTtext1_metrics=recipe_source["matched_native_GT_text1"],
                          matched_GTtext0p5_metrics=recipe_source["matched_native_GT_text0p5"],
                          benchmark_development_not_heldout=True)
    provenance = dict(recipe_source_sha256=digest(recipe_path.read_bytes()),
                      accepted_prediction_receipts_sha256=digest(receipts_path.read_bytes()),
                      native_result_report_sha256=recipe_source["native_score_report"]["SHA256"],
                      exact_decoded_caption_strings_preserved=True, no_caption_regeneration=True,
                      no_target_GT_caption_motion_or_RGB_exported=True,
                      prompt_contains_three_unrelated_training_style_examples=True,
                      raw_model_reasoning_and_responses_not_exported=True,
                      original_caption_JSON_byte_hashes_retained_per_case=True)
    readme = """# Best evaluated random20 generated captions

Contains1420 exact Q40 caption strings:20 windows for each of71 recordings.
Use video CFG1.5 and independent text CFG0.5 with the directional W1/no-camera
EMA200k model. Shape remains supplied. Feed text only to the motion expert;
the video-generator caption stays empty. Native UniPC35,97frames/20fps.

captions_by_id.json maps physical window ID to the exact conditioning string.
captions_random20.json and CSV include frozen frame bounds and motion/VLM seeds.
Q40_prompt.txt is the unchanged VLM instruction template. It is different from
the per-window generated captions. Do not substitute Q77 or PRO-B1 recaptions.
recipe.json contains actual generated/GT/control metrics and all model hashes.
PROVENANCE.json and per-case hashes bind these strings to accepted predictions.

No target GT caption, motion, RGB, checkpoint or credentials are included.
The prompt retains its three unrelated TRAINING style examples verbatim.
This repeatedly inspected benchmark is development data, not held-out proof.
Best observed MPJPE14.627242cm; video16.125472; GTtext1 13.073182.

Verify after download:
  python -m nymeria_baselines.vlm_caption_handoff verify --bundle-dir PATH

Model folder: https://drive.google.com/drive/folders/1f8KgvTkn-Ez_aZ3gSaS5Ty9C1PrBvJbp
EMA checkpoint: https://drive.google.com/file/d/1ow02esa0HQHeCk2I2M1b3l5MPTUJOHSC/view
Results: https://app.notion.com/p/415d2a44f44b80b2b687dd273fa22f36
"""
    directory.mkdir(parents=True, mode=0o775)
    payloads = {"captions_random20.json": json_bytes(data),
                "captions_by_id.json": json_bytes({r["id"]: r["caption"] for r in records}),
                "Q40_prompt.txt": prompt, "recipe.json": json_bytes(portable_recipe),
                "PROVENANCE.json": json_bytes(provenance), "README.md": readme.encode()}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    payloads["captions_random20.csv"] = buffer.getvalue().encode("utf-8")
    payloads["SHA256SUMS.json"] = json_bytes({name: digest(body) for name, body in sorted(payloads.items())})
    for name, body in payloads.items():
        with (directory / name).open("xb") as handle:
            handle.write(body)
        (directory / name).chmod(0o664)
    result = verify_bundle(directory)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for name, body in sorted(payloads.items()):
            info = zipfile.ZipInfo(directory.name + "/" + name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100664 << 16
            handle.writestr(info, body)
    archive.chmod(0o664)
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text(digest(archive.read_bytes()) + "  " + archive.name + "\n")
    checksum.chmod(0o664)
    return dict(result, archive=str(archive), archive_sha256=digest(archive.read_bytes()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--recipe", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--cases", type=Path)
    export.add_argument("--captions-root", type=Path)
    export.add_argument("--receipts", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_bundle(args.recipe, args.output_dir, cases_path=args.cases,
                               captions_root=args.captions_root, receipts_path=args.receipts)
    else:
        result = verify_bundle(args.bundle_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
