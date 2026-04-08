#!/usr/bin/env python3

import argparse
import json
import math
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


@dataclass
class CameraEntry:
    pano_id: str
    image_name: str
    color_path: Path
    depth_path: Path
    fx: float
    fy: float
    cx: float
    cy: float
    rot: np.ndarray
    trans: np.ndarray


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


def resolve_episode(args):
    if args.trajectory:
        if not args.scan:
            raise ValueError("--scan is required when using --trajectory")
        return args.scan, [vp.strip() for vp in args.trajectory.split(",") if vp.strip()], args.instr_id or "manual"

    if not (args.preds and args.annotations and args.instr_id):
        return None, None, None

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


def parse_camera_conf(scan_dir: Path, scan: str):
    conf_path = scan_dir / "undistorted_camera_parameters" / f"{scan}.conf"
    entries = []
    current_intr = None

    with open(conf_path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("intrinsics_matrix"):
                vals = line.split()[1:]
                current_intr = np.array([float(x) for x in vals], dtype=np.float32).reshape(3, 3)
                continue
            if not line.startswith("scan "):
                continue
            if current_intr is None:
                raise RuntimeError(f"encountered scan line before intrinsics in {conf_path}")

            tokens = line.split()
            depth_name = tokens[1]
            color_name = tokens[2]
            t = np.loadtxt(StringIO(" ".join(tokens[3:])), dtype=np.float32).reshape(4, 4)
            rot = t[:3, :3]
            trans = t[:3, 3]
            pano_id = color_name.split("_i")[0]
            entries.append(
                CameraEntry(
                    pano_id=pano_id,
                    image_name=color_name.replace(".jpg", ""),
                    color_path=scan_dir / "undistorted_color_images" / color_name,
                    depth_path=scan_dir / "undistorted_depth_images" / depth_name,
                    fx=float(current_intr[0, 0]),
                    fy=float(current_intr[1, 1]),
                    cx=float(current_intr[0, 2]),
                    cy=float(current_intr[1, 2]),
                    rot=rot,
                    trans=trans,
                )
            )
    return entries


def load_connectivity_positions(connectivity_dir: Path, scan: str):
    data = load_json(connectivity_dir / f"{scan}_connectivity.json")
    pos3 = {}
    graph = []
    for i, item in enumerate(data):
        if not item["included"]:
            continue
        vp = item["image_id"]
        pos3[vp] = np.array([item["pose"][3], item["pose"][7], item["pose"][11]], dtype=np.float32)
        for j, conn in enumerate(item["unobstructed"]):
            if conn and data[j]["included"] and i < j:
                graph.append((vp, data[j]["image_id"]))
    return pos3, graph


def project_entry_world_points(entry: CameraEntry, sample_step: int):
    depth = cv2.imread(str(entry.depth_path), cv2.IMREAD_UNCHANGED)
    color = cv2.imread(str(entry.color_path), cv2.IMREAD_COLOR)
    if depth is None or color is None:
        raise RuntimeError(f"failed to load {entry.depth_path} or {entry.color_path}")

    ys = np.arange(0, depth.shape[0], sample_step, dtype=np.float32)
    xs = np.arange(0, depth.shape[1], sample_step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    depth_s = depth[::sample_step, ::sample_step].astype(np.float32) / 4000.0
    color_s = color[::sample_step, ::sample_step].reshape(-1, 3)
    u = grid_x.reshape(-1)
    v = grid_y.reshape(-1)
    d = depth_s.reshape(-1)

    valid = (d > 0.1) & np.isfinite(d) & (d < 12.0)
    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    u = u[valid]
    v = v[valid]
    d = d[valid]
    rgb = color_s[valid]

    # Matterport undistorted cameras use x=right, y=up, -z=look.
    x_cam = (u - entry.cx) / entry.fx * d
    y_cam = -(v - entry.cy) / entry.fy * d
    z_cam = -d
    cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    world = cam @ entry.rot.T + entry.trans[None, :]
    return world.astype(np.float32), rgb.astype(np.uint8)


def compute_bounds(entries, eye_height, sample_step, max_height_above_eye):
    min_x = min_y = np.inf
    max_x = max_y = -np.inf
    z_cap = eye_height + max_height_above_eye

    for idx, entry in enumerate(entries, start=1):
        world, _ = project_entry_world_points(entry, sample_step)
        if len(world) == 0:
            continue
        world = world[world[:, 2] <= z_cap]
        if len(world) == 0:
            continue
        min_x = min(min_x, float(world[:, 0].min()))
        max_x = max(max_x, float(world[:, 0].max()))
        min_y = min(min_y, float(world[:, 1].min()))
        max_y = max(max_y, float(world[:, 1].max()))
        if idx % 120 == 0 or idx == len(entries):
            print(f"[bounds] processed {idx}/{len(entries)} cameras", flush=True)

    if not np.isfinite(min_x):
        raise RuntimeError("no valid points found while computing bounds")
    return min_x, max_x, min_y, max_y


def rasterize_bev(entries, eye_height, bounds, meters_per_px, sample_step, max_height_above_eye):
    min_x, max_x, min_y, max_y = bounds
    width = max(int(math.ceil((max_x - min_x) / meters_per_px)) + 1, 8)
    height = max(int(math.ceil((max_y - min_y) / meters_per_px)) + 1, 8)

    sum_rgb = np.zeros((height, width, 3), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float32)
    max_z = np.full((height, width), -1e9, dtype=np.float32)
    z_cap = eye_height + max_height_above_eye

    for idx, entry in enumerate(entries, start=1):
        world, rgb = project_entry_world_points(entry, sample_step)
        if len(world) == 0:
            continue

        valid = world[:, 2] <= z_cap
        world = world[valid]
        rgb = rgb[valid]
        if len(world) == 0:
            continue

        px = np.floor((world[:, 0] - min_x) / meters_per_px).astype(np.int32)
        py = np.floor((max_y - world[:, 1]) / meters_per_px).astype(np.int32)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        px = px[inside]
        py = py[inside]
        rgb = rgb[inside]
        wz = world[:, 2][inside]
        if len(px) == 0:
            continue

        # Prefer the highest non-ceiling surface per cell to keep furniture and room outlines visible.
        flat_idx = py * width + px
        order = np.lexsort((wz, flat_idx))
        flat_idx = flat_idx[order]
        wz = wz[order]
        rgb = rgb[order]

        last_flat = None
        best_rgb = None
        best_z = None
        best_count = 0.0
        for fi, z, color in zip(flat_idx, wz, rgb):
            if last_flat is None or fi != last_flat:
                if last_flat is not None:
                    y = last_flat // width
                    x = last_flat % width
                    max_z[y, x] = best_z
                    sum_rgb[y, x] += best_rgb
                    count[y, x] += best_count
                last_flat = fi
                best_z = float(z)
                best_rgb = color.astype(np.float64)
                best_count = 1.0
            else:
                if z >= best_z - 0.05:
                    best_rgb += color.astype(np.float64)
                    best_count += 1.0
                    best_z = max(best_z, float(z))
        if last_flat is not None:
            y = last_flat // width
            x = last_flat % width
            max_z[y, x] = best_z
            sum_rgb[y, x] += best_rgb
            count[y, x] += best_count
        if idx % 120 == 0 or idx == len(entries):
            print(f"[raster] processed {idx}/{len(entries)} cameras", flush=True)

    mask = count > 0
    bev = np.full((height, width, 3), 245, dtype=np.uint8)
    if np.any(mask):
        avg = np.zeros_like(sum_rgb, dtype=np.float32)
        avg[mask] = (sum_rgb[mask] / count[mask, None]).astype(np.float32)
        bev[mask] = np.clip(avg[mask], 0, 255).astype(np.uint8)

        # Very light cleanup: close tiny holes without heavily altering the raw point-cloud look.
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(bev, kernel, iterations=1)
        mask_u8 = mask.astype(np.uint8) * 255
        closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        tiny_fill = (closed > 0) & (~mask)
        bev[tiny_fill] = dilated[tiny_fill]

    return bev, mask, (width, height)


def stylize_bev(
    bev_rgb,
    mask,
    background_gray=236,
    solidify_kernel=5,
    solidify_iters=1,
    contrast=1.18,
    color_boost=1.08,
    brightness=0.95,
    sharpness=1.4,
    red_gain=1.03,
    green_gain=1.00,
    blue_gain=0.88,
):
    styled = bev_rgb.copy()
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((solidify_kernel, solidify_kernel), np.uint8)
    expanded = cv2.dilate(mask_u8, kernel, iterations=solidify_iters) > 0
    blur = cv2.GaussianBlur(styled, (0, 0), 1.1)
    new_fill = expanded & (~mask)
    styled[new_fill] = blur[new_fill]
    styled[~expanded] = background_gray

    image = Image.fromarray(styled)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color_boost)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int((sharpness - 1.0) * 220), threshold=2))
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    gains = np.array([red_gain, green_gain, blue_gain], dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(rgb * gains, 0, 255).astype(np.uint8)
    return rgb, expanded


def world_to_px(p, bounds, meters_per_px):
    min_x, max_x, min_y, max_y = bounds
    x = int(round((float(p[0]) - min_x) / meters_per_px))
    y = int(round((max_y - float(p[1])) / meters_per_px))
    return x, y


def overlay_topo(bev_rgb, pos3, edges, bounds, meters_per_px, line_width=2, node_radius=3):
    image = Image.fromarray(bev_rgb).convert("RGBA")
    draw = ImageDraw.Draw(image)

    for a, b in edges:
        if a not in pos3 or b not in pos3:
            continue
        draw.line(
            [world_to_px(pos3[a], bounds, meters_per_px), world_to_px(pos3[b], bounds, meters_per_px)],
            fill=(225, 228, 232, 215),
            width=line_width,
        )

    for vp, p in pos3.items():
        x, y = world_to_px(p, bounds, meters_per_px)
        r = node_radius
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(185, 190, 198, 225))
    return image.convert("RGB")


def add_caption(image, lines):
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    line_h = 22
    pad = 12
    box_h = pad * 2 + line_h * len(lines)
    odraw.rounded_rectangle((16, 16, 460, 16 + box_h), radius=14, fill=(18, 18, 18, 150))
    out = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(out)
    y = 26
    for line in lines:
        draw.text((28, y), line, fill=(255, 255, 255, 235))
        y += line_h
    return out.convert("RGB")


def save_gif(frames_rgb, out_path, fps):
    duration_ms = max(int(1000 / max(fps, 1)), 1)
    frames_rgb[0].save(
        out_path,
        save_all=True,
        append_images=frames_rgb[1:],
        duration=duration_ms,
        loop=0,
    )


def save_mp4(frames_rgb, out_path, fps):
    rgb0 = np.array(frames_rgb[0].convert("RGB"))
    h, w = rgb0.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames_rgb:
        bgr = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def save_contact_sheet(frames_rgb, labels, out_path, cols=3, thumb_w=400):
    thumbs = []
    for frame, label in zip(frames_rgb, labels):
        frame = frame.convert("RGB")
        thumb_h = int(frame.height * thumb_w / frame.width)
        thumb = frame.resize((thumb_w, thumb_h), getattr(Image, "Resampling", Image).LANCZOS)
        thumb = add_caption(thumb, [label])
        thumbs.append(thumb)

    rows = math.ceil(len(thumbs) / cols)
    thumb_h = thumbs[0].height
    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        canvas.paste(thumb, (x, y))
    canvas.save(out_path)


def draw_star(draw, center, radius, color, outline=None):
    cx, cy = center
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    draw.polygon(pts, fill=color, outline=outline)


def render_trajectory_frames(
    base_topo_rgb,
    pos3,
    trajectory,
    bounds,
    meters_per_px,
    episode_dir: Path,
    instr_id,
    scan,
    fps,
    traj_line_width=5,
    current_radius=9,
    goal_star_radius=14,
):
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    labels = []
    goal_vp = trajectory[-1]

    for step_idx, current_vp in enumerate(trajectory):
        frame = base_topo_rgb.copy().convert("RGBA")
        draw = ImageDraw.Draw(frame)

        path_so_far = trajectory[: step_idx + 1]
        pts = [world_to_px(pos3[vp], bounds, meters_per_px) for vp in path_so_far if vp in pos3]
        if len(pts) >= 2:
            draw.line(pts, fill=(226, 74, 51, 230), width=traj_line_width, joint="curve")

        start_pt = world_to_px(pos3[trajectory[0]], bounds, meters_per_px)
        draw.ellipse(
            (start_pt[0] - 10, start_pt[1] - 10, start_pt[0] + 10, start_pt[1] + 10),
            fill=(66, 133, 244, 240),
        )
        if goal_vp in pos3:
            goal_pt = world_to_px(pos3[goal_vp], bounds, meters_per_px)
            draw_star(draw, goal_pt, goal_star_radius, (52, 168, 83, 245), outline=None)

        if current_vp in pos3:
            curr_pt = world_to_px(pos3[current_vp], bounds, meters_per_px)
            draw.ellipse(
                (curr_pt[0] - current_radius, curr_pt[1] - current_radius, curr_pt[0] + current_radius, curr_pt[1] + current_radius),
                fill=(255, 183, 0, 240),
                outline=(120, 76, 0, 240),
                width=2,
            )

        next_vp = trajectory[step_idx + 1] if step_idx + 1 < len(trajectory) else None
        if next_vp and current_vp in pos3 and next_vp in pos3:
            seg = [
                world_to_px(pos3[current_vp], bounds, meters_per_px),
                world_to_px(pos3[next_vp], bounds, meters_per_px),
            ]
            draw.line(seg, fill=(156, 39, 176, 235), width=max(3, traj_line_width - 1))

        frame = add_caption(
            frame.convert("RGB"),
            [
                f"scan: {scan}",
                f"instr_id: {instr_id}",
                f"step: {step_idx + 1}/{len(trajectory)}",
                f"viewpoint: {current_vp}",
                f"next: {next_vp or 'STOP'}",
            ],
        )
        frame_path = frames_dir / f"frame_{step_idx:03d}.png"
        frame.save(frame_path)
        frames.append(frame)
        labels.append(f"step {step_idx + 1}: {current_vp}")

    save_gif(frames, episode_dir / "trajectory.gif", fps)
    save_mp4(frames, episode_dir / "trajectory.mp4", fps)
    save_contact_sheet(frames, labels, episode_dir / "contact_sheet.png")
    return {
        "frames_dir": str(frames_dir),
        "gif": str(episode_dir / "trajectory.gif"),
        "mp4": str(episode_dir / "trajectory.mp4"),
        "contact_sheet": str(episode_dir / "contact_sheet.png"),
    }


def save_side_by_side(left_path: Path, right_path: Path, out_path: Path):
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    target_h = max(left.height, right.height)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    if left.height != target_h:
        left = left.resize((int(left.width * target_h / left.height), target_h), resample)
    if right.height != target_h:
        right = right.resize((int(right.width * target_h / right.height), target_h), resample)
    canvas = Image.new("RGB", (left.width + right.width, target_h), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Build an orthographic RGB BEV map from Matterport undistorted RGB-D images.")
    parser.add_argument("--scan")
    parser.add_argument("--scans-root", required=True)
    parser.add_argument("--connectivity-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--meters-per-px", type=float, default=0.02)
    parser.add_argument("--sample-step", type=int, default=16)
    parser.add_argument("--max-height-above-eye", type=float, default=0.45)
    parser.add_argument("--topo-line-width", type=int, default=2)
    parser.add_argument("--topo-node-radius", type=int, default=3)
    parser.add_argument("--preds")
    parser.add_argument("--annotations")
    parser.add_argument("--instr-id")
    parser.add_argument("--trajectory", help="Comma-separated viewpoint ids for manual trajectory overlay")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--traj-line-width", type=int, default=5)
    parser.add_argument("--current-radius", type=int, default=9)
    parser.add_argument("--background-gray", type=int, default=236)
    parser.add_argument("--solidify-kernel", type=int, default=5)
    parser.add_argument("--solidify-iters", type=int, default=1)
    parser.add_argument("--contrast", type=float, default=1.18)
    parser.add_argument("--color-boost", type=float, default=1.08)
    parser.add_argument("--brightness", type=float, default=0.95)
    parser.add_argument("--sharpness", type=float, default=1.4)
    parser.add_argument("--red-gain", type=float, default=1.03)
    parser.add_argument("--green-gain", type=float, default=1.00)
    parser.add_argument("--blue-gain", type=float, default=0.88)
    parser.add_argument(
        "--left-image",
        "--compare-to",
        dest="compare_to",
        default=None,
        help="Optional left-panel image for compare_side_by_side.png. The current overlay is placed on the right.",
    )
    args = parser.parse_args()

    resolved_scan, trajectory, episode_name = resolve_episode(args)
    scan = resolved_scan or args.scan
    if not scan:
        raise ValueError("either --scan or (--preds --annotations --instr-id) is required")

    scan_dir = Path(args.scans_root) / scan
    out_dir = Path(args.output_dir) / scan
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] scan={scan}", flush=True)
    entries = parse_camera_conf(scan_dir, scan)
    pos3, edges = load_connectivity_positions(Path(args.connectivity_dir), scan)
    eye_height = float(np.median([p[2] for p in pos3.values()]))

    print(f"[build] cameras={len(entries)} | meters_per_px={args.meters_per_px} | sample_step={args.sample_step}", flush=True)
    bounds = compute_bounds(entries, eye_height, args.sample_step, args.max_height_above_eye)
    bev_rgb, mask, size = rasterize_bev(
        entries,
        eye_height,
        bounds,
        args.meters_per_px,
        args.sample_step,
        args.max_height_above_eye,
    )
    bev_rgb, styled_mask = stylize_bev(
        bev_rgb,
        mask,
        background_gray=args.background_gray,
        solidify_kernel=args.solidify_kernel,
        solidify_iters=args.solidify_iters,
        contrast=args.contrast,
        color_boost=args.color_boost,
        brightness=args.brightness,
        sharpness=args.sharpness,
        red_gain=args.red_gain,
        green_gain=args.green_gain,
        blue_gain=args.blue_gain,
    )
    overlay_rgb = overlay_topo(
        bev_rgb,
        pos3,
        edges,
        bounds,
        args.meters_per_px,
        line_width=args.topo_line_width,
        node_radius=args.topo_node_radius,
    )

    bev_path = out_dir / "rgb_bev.png"
    overlay_path = out_dir / "rgb_bev_topo.png"
    Image.fromarray(bev_rgb).save(bev_path)
    overlay_rgb.save(overlay_path)

    meta = {
        "scan": scan,
        "meters_per_px": args.meters_per_px,
        "sample_step": args.sample_step,
        "max_height_above_eye": args.max_height_above_eye,
        "topo_line_width": args.topo_line_width,
        "topo_node_radius": args.topo_node_radius,
        "background_gray": args.background_gray,
        "solidify_kernel": args.solidify_kernel,
        "solidify_iters": args.solidify_iters,
        "contrast": args.contrast,
        "color_boost": args.color_boost,
        "brightness": args.brightness,
        "sharpness": args.sharpness,
        "red_gain": args.red_gain,
        "green_gain": args.green_gain,
        "blue_gain": args.blue_gain,
        "eye_height_median": eye_height,
        "bounds": {
            "min_x": bounds[0],
            "max_x": bounds[1],
            "min_y": bounds[2],
            "max_y": bounds[3],
        },
        "image_size": {"width": size[0], "height": size[1]},
        "entries": len(entries),
        "bev": str(bev_path),
        "overlay": str(overlay_path),
    }

    if trajectory:
        episode_dir = out_dir / "trajectories" / episode_name
        print(f"[trajectory] rendering {episode_name} with {len(trajectory)} viewpoints", flush=True)
        traj_meta = render_trajectory_frames(
            overlay_rgb,
            pos3,
            trajectory,
            bounds,
            args.meters_per_px,
            episode_dir,
            episode_name,
            scan,
            args.fps,
            traj_line_width=args.traj_line_width,
            current_radius=args.current_radius,
        )
        meta["trajectory"] = {
            "instr_id": episode_name,
            "viewpoints": trajectory,
            **traj_meta,
        }

    if args.compare_to:
        compare_out = out_dir / "compare_side_by_side.png"
        save_side_by_side(Path(args.compare_to), overlay_path, compare_out)
        meta["compare"] = str(compare_out)

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
