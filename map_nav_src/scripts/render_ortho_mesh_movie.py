#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path

# Prefer EGL for GPU-backed offscreen rendering. Mesa/OSMesa remains available
# as a system fallback if EGL cannot initialize on a given machine.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageEnhance

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_rgb_ortho_bev import overlay_topo, render_trajectory_frames  # noqa: E402
from graph_nav_movie import (  # noqa: E402
    accessor_array,
    ensure_decoded_glb_textures,
    ensure_dir,
    expand_r2r_annotations,
    expand_reverie_annotations,
    flatten_pred_trajectory,
    load_glb,
    load_graph_for_scan,
    load_json,
    load_texture_arrays,
    resolve_material_texture_info,
)


def rotation_from_forward_up(forward, up):
    forward = np.asarray(forward, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    up = up / (np.linalg.norm(up) + 1e-8)
    z_axis = -forward
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    return pose


def primitive_to_trimesh(gltf, buf, prim, material_info, texture_arrays):
    attrs = prim.get("attributes", {})
    if "POSITION" not in attrs or "indices" not in prim:
        return None

    vertices = accessor_array(gltf, buf, attrs["POSITION"]).astype(np.float32)
    indices = accessor_array(gltf, buf, prim["indices"]).reshape(-1).astype(np.int64)
    if len(indices) < 3:
        return None
    faces = indices[: len(indices) // 3 * 3].reshape(-1, 3)

    normals = None
    if "NORMAL" in attrs:
        normals = accessor_array(gltf, buf, attrs["NORMAL"]).astype(np.float32)

    uv = None
    if "TEXCOORD_0" in attrs:
        uv = accessor_array(gltf, buf, attrs["TEXCOORD_0"]).astype(np.float32)

    material_meta = material_info.get(prim.get("material"), {"image_idx": -1, "base_rgb": (255, 255, 255)})
    base_rgba = list(material_meta["base_rgb"]) + [255]
    texture_index = int(material_meta["image_idx"])

    if uv is not None and texture_index in texture_arrays:
        texture_image = Image.fromarray(texture_arrays[texture_index].astype(np.uint8), mode="RGB")
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture_image,
            baseColorFactor=base_rgba,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            emissiveFactor=[0.0, 0.0, 0.0],
        )
        visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    else:
        colors = np.repeat(np.asarray(base_rgba, dtype=np.uint8)[None, :], len(vertices), axis=0)
        visual = trimesh.visual.ColorVisuals(vertex_colors=colors)

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        visual=visual,
        process=False,
        validate=False,
    )


def load_textured_scene(mesh_path, ambient_intensity=0.08):
    gltf, buf = load_glb(mesh_path)
    material_info = resolve_material_texture_info(gltf)
    texture_arrays = load_texture_arrays(ensure_decoded_glb_textures(mesh_path))

    scene = pyrender.Scene(
        ambient_light=np.array([ambient_intensity, ambient_intensity, ambient_intensity, 1.0], dtype=np.float32),
        bg_color=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )
    all_vertices = []
    mesh_count = 0
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            tri_mesh = primitive_to_trimesh(gltf, buf, prim, material_info, texture_arrays)
            if tri_mesh is None or tri_mesh.faces is None or len(tri_mesh.faces) == 0:
                continue
            scene.add(pyrender.Mesh.from_trimesh(tri_mesh, smooth=False))
            all_vertices.append(np.asarray(tri_mesh.vertices, dtype=np.float32))
            mesh_count += 1

    if not all_vertices:
        raise RuntimeError(f"no renderable primitives found in {mesh_path}")

    vertices = np.concatenate(all_vertices, axis=0)
    world_min = vertices.min(axis=0)
    world_max = vertices.max(axis=0)
    return scene, world_min, world_max, mesh_count, len(texture_arrays)


def add_default_lighting(scene, center, world_top_z, key_light_intensity=5.0, fill_light_intensity=2.3):
    light_specs = [
        ((0.55, 0.25, -1.0), key_light_intensity),
        ((-0.25, 0.55, -1.0), fill_light_intensity),
    ]
    for forward, intensity in light_specs:
        pose = rotation_from_forward_up(forward, (0.0, 1.0, 0.0))
        pose[:3, 3] = np.array([center[0], center[1], world_top_z + 6.0], dtype=np.float32)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=float(intensity)), pose=pose)


def render_topdown_ortho(
    mesh_path,
    width=2200,
    padding_frac=0.02,
    post_contrast=1.08,
    post_sharpness=1.12,
    ambient_intensity=0.08,
    key_light_intensity=5.0,
    fill_light_intensity=2.3,
):
    scene, world_min, world_max, mesh_count, texture_count = load_textured_scene(
        mesh_path,
        ambient_intensity=ambient_intensity,
    )
    center = (world_min + world_max) / 2.0
    span = np.maximum(world_max - world_min, 1e-3)

    xmag = float(span[0] * (1.0 + padding_frac) / 2.0)
    ymag = float(span[1] * (1.0 + padding_frac) / 2.0)
    meters_per_px = (2.0 * xmag) / max(width, 1)
    height = max(int(round((2.0 * ymag) / meters_per_px)), 1)

    camera = pyrender.OrthographicCamera(
        xmag=xmag,
        ymag=ymag,
        znear=0.05,
        zfar=float(max(50.0, span[2] + 30.0)),
    )
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, 3] = np.array([center[0], center[1], world_max[2] + 10.0], dtype=np.float32)
    scene.add(camera, pose=camera_pose)
    add_default_lighting(
        scene,
        center,
        float(world_max[2]),
        key_light_intensity=key_light_intensity,
        fill_light_intensity=fill_light_intensity,
    )

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, depth = renderer.render(
            scene,
            flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL,
        )
    finally:
        renderer.delete()

    image = Image.fromarray(color, mode="RGBA").convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(post_contrast)
    image = ImageEnhance.Sharpness(image).enhance(post_sharpness)

    world_bounds = (
        float(center[0] - xmag),
        float(center[0] + xmag),
        float(center[1] - ymag),
        float(center[1] + ymag),
    )
    return {
        "bev_rgb": np.asarray(image, dtype=np.uint8),
        "depth": depth,
        "meters_per_px": meters_per_px,
        "bounds": world_bounds,
        "world_min": world_min.tolist(),
        "world_max": world_max.tolist(),
        "mesh_count": mesh_count,
        "texture_count": texture_count,
        "image_size": [width, height],
    }


def build_annotation_lookup(dataset, annotations_path):
    if dataset == "r2r":
        return expand_r2r_annotations(annotations_path)
    if dataset == "reverie":
        return expand_reverie_annotations(annotations_path)
    raise ValueError(f"unsupported dataset: {dataset}")


def render_episode(args):
    preds = load_json(args.preds)
    anno_lookup = build_annotation_lookup(args.dataset, args.annotations)
    preds = [p for p in preds if p.get("instr_id") in anno_lookup]
    if args.instr_id:
        preds = [p for p in preds if p.get("instr_id") == args.instr_id]
    if not preds:
        raise KeyError(f"instr_id {args.instr_id!r} not found after annotation filtering")

    pred_item = preds[0]
    instr_id = pred_item["instr_id"]
    meta = anno_lookup[instr_id]
    scan = meta["scan"]

    graph, pos3 = load_graph_for_scan(args.connectivity_dir, scan)
    trajectory = flatten_pred_trajectory(pred_item["trajectory"])
    if not trajectory:
        raise RuntimeError(f"prediction for {instr_id} has empty trajectory")

    mesh_path = Path(args.mesh_dir) / scan / f"{scan}.glb"
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    render = render_topdown_ortho(
        str(mesh_path),
        width=args.width,
        padding_frac=args.padding_frac,
        post_contrast=args.post_contrast,
        post_sharpness=args.post_sharpness,
        ambient_intensity=args.ambient_intensity,
        key_light_intensity=args.key_light_intensity,
        fill_light_intensity=args.fill_light_intensity,
    )

    out_root = Path(args.output_dir) / scan
    ensure_dir(out_root)
    bev_path = out_root / "rgb_bev.png"
    topo_path = out_root / "rgb_bev_topo.png"
    scan_meta_path = out_root / "meta.json"

    bev_image = Image.fromarray(render["bev_rgb"])
    bev_image.save(bev_path)
    topo_image = overlay_topo(
        render["bev_rgb"],
        pos3,
        graph.edges(),
        render["bounds"],
        render["meters_per_px"],
        line_width=args.topo_line_width,
        node_radius=args.topo_node_radius,
    )
    topo_image.save(topo_path)

    episode_dir = out_root / "trajectories" / instr_id
    traj_meta = render_trajectory_frames(
        topo_image,
        pos3,
        trajectory,
        render["bounds"],
        render["meters_per_px"],
        episode_dir=episode_dir,
        instr_id=instr_id,
        scan=scan,
        fps=args.fps,
        traj_line_width=args.traj_line_width,
        current_radius=args.current_radius,
        goal_star_radius=args.goal_star_radius,
    )

    meta_payload = {
        "dataset": args.dataset,
        "scan": scan,
        "instr_id": instr_id,
        "mesh_path": str(mesh_path),
        "prediction_path": args.preds,
        "annotation_path": args.annotations,
        "trajectory": trajectory,
        "world_bounds": render["bounds"],
        "meters_per_px": render["meters_per_px"],
        "image_size": render["image_size"],
        "mesh_count": render["mesh_count"],
        "texture_count": render["texture_count"],
        "ambient_intensity": args.ambient_intensity,
        "key_light_intensity": args.key_light_intensity,
        "fill_light_intensity": args.fill_light_intensity,
        "outputs": {
            "rgb_bev": str(bev_path),
            "rgb_bev_topo": str(topo_path),
            **traj_meta,
        },
    }
    with open(scan_meta_path, "w") as f:
        json.dump(
            {
                "scan": scan,
                "mesh_path": str(mesh_path),
                "world_bounds": render["bounds"],
                "meters_per_px": render["meters_per_px"],
                "image_size": render["image_size"],
                "mesh_count": render["mesh_count"],
                "texture_count": render["texture_count"],
                "ambient_intensity": args.ambient_intensity,
                "key_light_intensity": args.key_light_intensity,
                "fill_light_intensity": args.fill_light_intensity,
                "rgb_bev": str(bev_path),
                "rgb_bev_topo": str(topo_path),
            },
            f,
            indent=2,
        )
    with open(episode_dir / "meta.json", "w") as f:
        json.dump(meta_payload, f, indent=2)
    return meta_payload


def main():
    parser = argparse.ArgumentParser(description="Render a true 3D orthographic Matterport mesh top-view and overlay topo trajectories.")
    parser.add_argument("--dataset", choices=["r2r", "reverie"], required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--connectivity-dir", required=True)
    parser.add_argument("--mesh-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--instr-id", required=True)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--width", type=int, default=2200)
    parser.add_argument("--padding-frac", type=float, default=0.02)
    parser.add_argument("--post-contrast", type=float, default=1.08)
    parser.add_argument("--post-sharpness", type=float, default=1.12)
    parser.add_argument("--ambient-intensity", type=float, default=0.08)
    parser.add_argument("--key-light-intensity", type=float, default=5.0)
    parser.add_argument("--fill-light-intensity", type=float, default=2.3)
    parser.add_argument("--topo-line-width", type=int, default=7)
    parser.add_argument("--topo-node-radius", type=int, default=8)
    parser.add_argument("--traj-line-width", type=int, default=11)
    parser.add_argument("--current-radius", type=int, default=9)
    parser.add_argument("--goal-star-radius", type=int, default=20)
    args = parser.parse_args()

    meta_payload = render_episode(args)
    print(json.dumps(meta_payload["outputs"], indent=2))


if __name__ == "__main__":
    main()
