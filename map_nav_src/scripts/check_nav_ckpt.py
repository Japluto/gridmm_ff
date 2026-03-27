#!/usr/bin/env python3

import argparse
import os
from typing import Any, Dict, Tuple

import torch


def _safe_shape(value: Any):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    return None


def _describe_resume_block(name: str, block: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(block, dict):
        return False, f"{name}: not a dict"
    if "state_dict" not in block:
        return False, f"{name}: missing state_dict"
    if not isinstance(block["state_dict"], dict):
        return False, f"{name}: state_dict is not a dict"
    return True, f"{name}: state_dict_keys={len(block['state_dict'])}"


def _detect_text_backbone_shape(state_dict: Dict[str, Any]) -> str:
    key = "bert.embeddings.word_embeddings.weight"
    if key not in state_dict:
        return "unknown"
    shape = _safe_shape(state_dict[key])
    if shape is None:
        return "unknown"
    vocab_size = shape[0]
    if vocab_size == 30522:
        return "bert-base-like (30522 vocab)"
    if vocab_size == 250002:
        return "xlm-roberta-base-like (250002 vocab)"
    return f"unknown-vocab ({vocab_size})"


def inspect_checkpoint(path: str) -> int:
    obj = torch.load(path, map_location="cpu")
    print(f"path: {path}")
    print(f"python_type: {type(obj).__name__}")

    if not isinstance(obj, dict):
        print("result: unsupported top-level type; not usable as resume_file")
        return 1

    top_keys = list(obj.keys())
    print(f"top_level_keys_sample: {top_keys[:20]}")
    print(f"top_level_key_count: {len(top_keys)}")

    is_resume_like = "vln_bert" in obj and "critic" in obj
    if is_resume_like:
        ok_vln, msg_vln = _describe_resume_block("vln_bert", obj["vln_bert"])
        ok_critic, msg_critic = _describe_resume_block("critic", obj["critic"])
        print(msg_vln)
        print(msg_critic)
        if ok_vln:
            state_dict = obj["vln_bert"]["state_dict"]
            print(f"text_backbone_hint: {_detect_text_backbone_shape(state_dict)}")
        if ok_vln and ok_critic:
            print("result: looks like a full nav resume checkpoint; usable as --resume_file")
            return 0
        print("result: has resume-like keys but malformed blocks; likely not usable as --resume_file")
        return 1

    flat_state_dict = all(torch.is_tensor(v) for v in obj.values())
    if flat_state_dict:
        print("result: looks like a flat model state_dict / init checkpoint, not a full listener resume checkpoint")
        print(f"text_backbone_hint: {_detect_text_backbone_shape(obj)}")
        interesting_prefixes = [
            "bert.",
            "global_sap_head",
            "local_sap_head",
            "sap_fuse",
            "critic",
        ]
        for prefix in interesting_prefixes:
            count = sum(1 for k in obj.keys() if k.startswith(prefix))
            if count:
                print(f"prefix_count[{prefix}] = {count}")
        print("usable_as_resume_file: no")
        print("possible_use: initialization via --bert_ckpt_file if model/tokenizer branch matches")
        return 2

    print("result: dict checkpoint, but neither resume-file format nor flat state_dict")
    nested = {
        k: type(v).__name__
        for k, v in list(obj.items())[:20]
    }
    print(f"nested_key_types_sample: {nested}")
    return 1


def main():
    parser = argparse.ArgumentParser(description="Inspect a GridMM nav checkpoint.")
    parser.add_argument("checkpoint", help="Path to .pt/.pth checkpoint")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    raise SystemExit(inspect_checkpoint(ckpt_path))


if __name__ == "__main__":
    main()
