#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_json(path):
    with open(path) as f:
        return json.load(f)


def flatten_pred_trajectory(traj):
    out = []
    for step in traj:
        if isinstance(step, (list, tuple)):
            if len(step) == 0:
                continue
            if len(step) == 1 and isinstance(step[0], str):
                out.append(step[0])
            else:
                for vp in step:
                    if isinstance(vp, str):
                        out.append(vp)
        elif isinstance(step, str):
            out.append(step)
        elif isinstance(step, dict) and "path" in step:
            out.extend(flatten_pred_trajectory(step["path"]))
    return out


def load_annotation_lookup(annotations):
    lookup = {}
    for item in annotations:
        path_id = str(item["path_id"])
        instructions = item.get("instructions", [])
        for idx, _ in enumerate(instructions):
            lookup[f"{path_id}_{idx}"] = item
    return lookup


def load_scan_positions(connectivity_dir, scan):
    path = os.path.join(connectivity_dir, f"{scan}_connectivity.json")
    data = load_json(path)
    pos3 = {}
    for item in data:
        if not item["included"]:
            continue
        pos3[item["image_id"]] = np.array(
            [float(item["pose"][3]), float(item["pose"][7]), float(item["pose"][11])],
            dtype=np.float32,
        )
    return pos3


def normalize_angle(x):
    pi2 = 2 * math.pi
    x = x % pi2
    if x > math.pi:
        x -= pi2
    return x


def compute_heading_elevation(pos_a, pos_b):
    dx = float(pos_b[0] - pos_a[0])
    dy = float(pos_b[1] - pos_a[1])
    dz = float(pos_b[2] - pos_a[2])
    xy_dist = max(math.sqrt(dx * dx + dy * dy), 1e-8)
    xyz_dist = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-8)

    # Match the Matterport / GridMM heading convention.
    heading = math.asin(dx / xy_dist)
    if pos_b[1] < pos_a[1]:
        heading = math.pi - heading
    elevation = math.asin(dz / xyz_dist)
    return normalize_angle(heading), elevation


def load_faces(scan_dir, viewpoint_id):
    skybox_dir = Path(scan_dir) / "matterport_skybox_images"
    faces = []
    for face_idx in range(6):
        face_path = skybox_dir / f"{viewpoint_id}_skybox{face_idx}_sami.jpg"
        if not face_path.exists():
            break
        img = cv2.imread(str(face_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read {face_path}")
        faces.append(img)

    if len(faces) == 6:
        return faces

    merged_path = skybox_dir / f"{viewpoint_id}_skybox_small.jpg"
    if not merged_path.exists():
        raise FileNotFoundError(f"missing skybox images for viewpoint {viewpoint_id}")
    merged = cv2.imread(str(merged_path), cv2.IMREAD_COLOR)
    if merged is None:
        raise RuntimeError(f"failed to read {merged_path}")
    face_w = merged.shape[1] // 6
    return [merged[:, i * face_w : (i + 1) * face_w].copy() for i in range(6)]


class CubemapRenderer:
    def __init__(self, width, height, vfov_deg):
        self.width = width
        self.height = height
        self.vfov = math.radians(vfov_deg)
        self.hfov = 2.0 * math.atan(math.tan(self.vfov / 2.0) * width / height)

        xs = (2.0 * (np.arange(width, dtype=np.float32) + 0.5) / width - 1.0) * math.tan(self.hfov / 2.0)
        ys = (1.0 - 2.0 * (np.arange(height, dtype=np.float32) + 0.5) / height) * math.tan(self.vfov / 2.0)
        grid_x, grid_y = np.meshgrid(xs, ys)

        # Camera coordinates: +x right, +y forward, +z up.
        dirs = np.stack([grid_x, np.ones_like(grid_x), grid_y], axis=-1)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        self.base_dirs = dirs / np.maximum(norms, 1e-8)

    def render(self, faces, heading, elevation):
        chead = math.cos(heading)
        shead = math.sin(heading)
        celev = math.cos(elevation)
        selev = math.sin(elevation)

        rot_z = np.array(
            [[chead, -shead, 0.0], [shead, chead, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        rot_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, celev, -selev], [0.0, selev, celev]],
            dtype=np.float32,
        )
        rot = rot_z @ rot_x

        dirs = self.base_dirs @ rot.T
        x = dirs[..., 0]
        y = dirs[..., 1]
        z = dirs[..., 2]
        ax = np.abs(x)
        ay = np.abs(y)
        az = np.abs(z)

        output = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        face_specs = [
            ((az >= ax) & (az >= ay) & (z >= 0), 0, x / np.maximum(az, 1e-8), y / np.maximum(az, 1e-8)),   # up
            ((ay >= ax) & (ay >= az) & (y >= 0), 1, x / np.maximum(ay, 1e-8), -z / np.maximum(ay, 1e-8)),  # front
            ((ax >= ay) & (ax >= az) & (x >= 0), 2, -y / np.maximum(ax, 1e-8), -z / np.maximum(ax, 1e-8)), # right
            ((ay >= ax) & (ay >= az) & (y < 0), 3, -x / np.maximum(ay, 1e-8), -z / np.maximum(ay, 1e-8)),  # back
            ((ax >= ay) & (ax >= az) & (x < 0), 4, y / np.maximum(ax, 1e-8), -z / np.maximum(ax, 1e-8)),   # left
            ((az >= ax) & (az >= ay) & (z < 0), 5, x / np.maximum(az, 1e-8), -y / np.maximum(az, 1e-8)),   # down
        ]

        for mask, face_idx, u, v in face_specs:
            if not np.any(mask):
                continue
            face = faces[face_idx]
            h, w = face.shape[:2]
            map_x = ((u + 1.0) * 0.5 * (w - 1)).astype(np.float32)
            map_y = ((v + 1.0) * 0.5 * (h - 1)).astype(np.float32)
            sampled = cv2.remap(face, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            output[mask] = sampled[mask]

        return output


def add_caption_bgr(frame_bgr, lines):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    pad = 10
    line_h = 20
    box_h = pad * 2 + line_h * len(lines)
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle((10, 10, 480, 10 + box_h), fill=(20, 20, 20, 160))
    pil = Image.alpha_composite(pil.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(pil)
    y = 18
    for line in lines:
        draw.text((20, y), line, fill=(255, 255, 255))
        y += line_h
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def save_gif(frames_bgr, out_path, fps):
    frames_rgb = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames_bgr]
    duration_ms = max(int(1000 / max(fps, 1)), 1)
    frames_rgb[0].save(
        out_path,
        save_all=True,
        append_images=frames_rgb[1:],
        duration=duration_ms,
        loop=0,
    )


def save_mp4(frames_bgr, out_path, fps):
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()


def save_contact_sheet(frames_bgr, labels, out_path, cols=3, thumb_w=360):
    thumbs = []
    for frame, label in zip(frames_bgr, labels):
        h, w = frame.shape[:2]
        thumb_h = int(h * thumb_w / w)
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumb = add_caption_bgr(thumb, [label])
        thumbs.append(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)))

    rows = math.ceil(len(thumbs) / cols)
    thumb_h = thumbs[0].height
    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        canvas.paste(thumb, (x, y))
    canvas.save(out_path)


def resolve_episode(args):
    if args.scan and args.trajectory:
        return args.scan, [vp.strip() for vp in args.trajectory.split(",") if vp.strip()], args.instr_id or "manual"

    preds = load_json(args.preds)
    annotations = load_json(args.annotations)
    anno_lookup = load_annotation_lookup(annotations)
    pred_lookup = {item["instr_id"]: item for item in preds}

    if args.instr_id not in pred_lookup:
        raise KeyError(f"instr_id {args.instr_id} not found in preds")
    if args.instr_id not in anno_lookup:
        raise KeyError(f"instr_id {args.instr_id} not found in annotations")

    pred_item = pred_lookup[args.instr_id]
    anno_item = anno_lookup[args.instr_id]
    return anno_item["scan"], flatten_pred_trajectory(pred_item["trajectory"]), args.instr_id


def main():
    parser = argparse.ArgumentParser(description="Export first-person trajectory frames from Matterport skybox data.")
    parser.add_argument("--preds")
    parser.add_argument("--annotations")
    parser.add_argument("--instr-id")
    parser.add_argument("--scan")
    parser.add_argument("--trajectory", help="Comma-separated viewpoint ids for manual export")
    parser.add_argument("--connectivity-dir", required=True)
    parser.add_argument("--skybox-root", required=True, help="Directory containing scan folders with matterport_skybox_images")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--vfov", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()

    scan, trajectory, instr_id = resolve_episode(args)
    if not trajectory:
        raise ValueError("empty trajectory")

    pos3 = load_scan_positions(args.connectivity_dir, scan)
    scan_dir = Path(args.skybox_root) / scan
    if not scan_dir.exists():
        raise FileNotFoundError(f"scan directory not found: {scan_dir}")

    renderer = CubemapRenderer(args.width, args.height, args.vfov)

    episode_dir = Path(args.output_dir) / instr_id
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    headings = []
    elevations = []
    for i, vp in enumerate(trajectory):
        if i + 1 < len(trajectory) and trajectory[i + 1] in pos3 and vp in pos3:
            heading, elevation = compute_heading_elevation(pos3[vp], pos3[trajectory[i + 1]])
        elif headings:
            heading, elevation = headings[-1], elevations[-1]
        else:
            heading, elevation = 0.0, 0.0
        headings.append(heading)
        elevations.append(elevation)

    cache = {}
    frames = []
    labels = []
    for step_idx, vp in enumerate(trajectory):
        if vp not in cache:
            cache[vp] = load_faces(scan_dir, vp)
        frame = renderer.render(cache[vp], headings[step_idx], elevations[step_idx])
        frame = add_caption_bgr(
            frame,
            [
                f"instr_id: {instr_id}",
                f"scan: {scan}",
                f"step: {step_idx + 1}/{len(trajectory)}",
                f"viewpoint: {vp}",
            ],
        )
        frame_path = frames_dir / f"frame_{step_idx:03d}.png"
        cv2.imwrite(str(frame_path), frame)
        frames.append(frame)
        labels.append(f"step {step_idx + 1}: {vp}")

    save_gif(frames, str(episode_dir / "trajectory.gif"), args.fps)
    save_mp4(frames, str(episode_dir / "trajectory.mp4"), args.fps)
    save_contact_sheet(frames, labels, str(episode_dir / "contact_sheet.png"))

    meta = {
        "instr_id": instr_id,
        "scan": scan,
        "trajectory": trajectory,
        "headings": headings,
        "elevations": elevations,
        "frames_dir": str(frames_dir),
        "gif": str(episode_dir / "trajectory.gif"),
        "mp4": str(episode_dir / "trajectory.mp4"),
        "contact_sheet": str(episode_dir / "contact_sheet.png"),
    }
    with open(episode_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] {instr_id} -> {episode_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
