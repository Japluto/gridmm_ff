#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def flatten_trajectory(trajectory):
    fixed = []
    for step in trajectory:
        if not isinstance(step, (list, tuple)):
            raise AssertionError(f"Trajectory step must be list/tuple, got {type(step)!r}: {step!r}")
        if len(step) == 0:
            continue
        if len(step) == 1:
            if not isinstance(step[0], str):
                raise AssertionError(f"Step viewpoint must be str: {step!r}")
            fixed.append([step[0]])
        else:
            for vp in step:
                if not isinstance(vp, str):
                    raise AssertionError(f"Flattened viewpoint must be str: {step!r}")
                fixed.append([vp])
    return fixed


def validate_preds(preds):
    for item in preds:
        assert "instr_id" in item, item
        assert "trajectory" in item, item
        assert isinstance(item["trajectory"], list), (item["instr_id"], type(item["trajectory"]))
        for step in item["trajectory"]:
            assert isinstance(step, (list, tuple)), (item["instr_id"], step)
            assert len(step) == 1, (item["instr_id"], step)
            assert isinstance(step[0], str), (item["instr_id"], step)


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: fix_r2r_submission_trajectory.py INPUT_JSON [OUTPUT_JSON]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else input_path.with_name(input_path.stem + "_fixed.json")

    with input_path.open() as f:
        preds = json.load(f)

    fixed_preds = []
    bad_before = 0
    for item in preds:
        trajectory = item.get("trajectory", [])
        for step in trajectory:
            if isinstance(step, (list, tuple)) and len(step) > 1:
                bad_before += 1
        fixed_item = dict(item)
        fixed_item["trajectory"] = flatten_trajectory(trajectory)
        fixed_preds.append(fixed_item)

    validate_preds(fixed_preds)

    with output_path.open("w") as f:
        json.dump(fixed_preds, f, sort_keys=True, indent=4, separators=(",", ": "))

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"items={len(fixed_preds)}")
    print(f"bad_steps_before={bad_before}")
    print("bad_steps_after=0")


if __name__ == "__main__":
    main()
