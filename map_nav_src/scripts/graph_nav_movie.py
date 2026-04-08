#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import textwrap
import struct
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont


CANVAS_W = 1600
CANVAS_H = 900
GRAPH_BOX = (40, 40, 980, 860)
SIDEBAR_BOX = (1020, 40, 1560, 860)

GLTF_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
GLTF_TYPE_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
TEXTURE_CACHE_ROOT = Path(tempfile.gettempdir()) / "gridmm_mp3d_texture_cache"
NODE_MIN_MAJOR = 14
RESAMPLING = getattr(Image, "Resampling", Image)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_node_major_version(text):
    match = re.search(r"v?(\d+)", text.strip())
    return int(match.group(1)) if match else -1


def find_node_binary():
    candidates = []
    system_node = shutil.which("node")
    if system_node:
        candidates.append(Path(system_node))

    nvm_root = Path.home() / ".nvm" / "versions" / "node"
    if nvm_root.exists():
        candidates.extend(sorted(nvm_root.glob("v*/bin/node"), reverse=True))

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if not candidate.exists():
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            proc = subprocess.run(
                [str(candidate), "-v"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue
        if parse_node_major_version(proc.stdout) >= NODE_MIN_MAJOR:
            return str(candidate)
    return None


def ensure_basis_decoder_deps(node_bin):
    tool_root = Path(tempfile.gettempdir()) / "gridmm_basis_decoder"
    ensure_dir(tool_root)

    basis_pkg = tool_root / "node_modules" / "basis_universal_wasm"
    png_pkg = tool_root / "node_modules" / "pngjs"
    if basis_pkg.exists() and png_pkg.exists():
        return str(tool_root)

    node_path = Path(node_bin)
    npm_cli = node_path.parent.parent / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"
    npm_cmd = [node_bin, str(npm_cli)] if npm_cli.exists() else None
    if npm_cmd is None:
        npm_bin = str(node_path.with_name("npm"))
        if Path(npm_bin).exists():
            npm_cmd = [npm_bin]
        else:
            npm_bin = shutil.which("npm")
            if not npm_bin:
                raise RuntimeError("npm not found for Basis texture decoder")
            npm_cmd = [npm_bin]

    if not (tool_root / "package.json").exists():
        subprocess.run(
            npm_cmd + ["init", "-y"],
            cwd=str(tool_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    subprocess.run(
        npm_cmd + ["install", "basis_universal_wasm", "pngjs"],
        cwd=str(tool_root),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return str(tool_root)


def ensure_decoded_glb_textures(glb_path):
    node_bin = find_node_binary()
    if not node_bin:
        raise RuntimeError("Node.js >= 14 not found; cannot decode MP3D Basis textures")

    deps_dir = ensure_basis_decoder_deps(node_bin)
    glb_path = Path(glb_path)
    out_dir = TEXTURE_CACHE_ROOT / f"{glb_path.parent.name}_{glb_path.stem}"
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        return manifest_path

    ensure_dir(out_dir)
    helper_path = Path(__file__).with_name("decode_glb_basis_textures.js")
    proc = subprocess.run(
        [node_bin, str(helper_path), str(glb_path), str(out_dir), deps_dir],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest_text = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else str(manifest_path)
    return Path(manifest_text)


def load_texture_arrays(manifest_path):
    payload = load_json(manifest_path)
    texture_arrays = {}
    for item in payload.get("images", []):
        image_path = item.get("path")
        if not image_path or not os.path.exists(image_path):
            continue
        try:
            texture_arrays[int(item["index"])] = np.asarray(Image.open(image_path).convert("RGB"))
        except Exception:
            continue
    return texture_arrays


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
    return out


def load_graph_for_scan(connectivity_dir, scan):
    path = os.path.join(connectivity_dir, f"{scan}_connectivity.json")
    with open(path) as f:
        data = json.load(f)

    graph = nx.Graph()
    pos3 = {}
    for i, item in enumerate(data):
        if not item["included"]:
            continue
        pos3[item["image_id"]] = (
            float(item["pose"][3]),
            float(item["pose"][7]),
            float(item["pose"][11]),
        )
        for j, conn in enumerate(item["unobstructed"]):
            if conn and data[j]["included"]:
                a = item["image_id"]
                b = data[j]["image_id"]
                pa = pos3[a]
                pb = (
                    float(data[j]["pose"][3]),
                    float(data[j]["pose"][7]),
                    float(data[j]["pose"][11]),
                )
                dist = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2)
                graph.add_edge(a, b, weight=dist)
    return graph, pos3


def build_display_positions(pos3, width, height, margin=40, bounds=None):
    # Matterport3DSimulator uses z-up coordinates, so the navigation graph lives on x-y.
    xs = [p[0] for p in pos3.values()]
    ys = [p[1] for p in pos3.values()]
    if bounds is None:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x, max_x, min_y, max_y = bounds
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    disp = {}
    for vp, (x, y, _) in pos3.items():
        px = margin + (x - min_x) * scale
        py = height - margin - (y - min_y) * scale
        disp[vp] = (px, py)
    return disp, (min_x, max_x, min_y, max_y)


def pad_world_bounds(bounds, frac=0.03, min_pad=0.35):
    min_x, max_x, min_y, max_y = bounds
    pad_x = max((max_x - min_x) * frac, min_pad)
    pad_y = max((max_y - min_y) * frac, min_pad)
    return (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)


def load_glb(path):
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"invalid glb magic for {path}")
    off = 12
    json_chunk = None
    bin_chunk = None
    while off < length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off : off + chunk_len]
        off += chunk_len
        if chunk_type == 0x4E4F534A:
            json_chunk = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\x00"))
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk
    if json_chunk is None or bin_chunk is None:
        raise ValueError(f"incomplete glb {path}")
    return json_chunk, bin_chunk


def accessor_array(gltf, buf, idx):
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    dtype = GLTF_COMPONENT_DTYPE[acc["componentType"]]
    comps = GLTF_TYPE_COUNTS[acc["type"]]
    item_size = np.dtype(dtype).itemsize * comps
    offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    stride = bv.get("byteStride", item_size)
    if stride == item_size:
        return np.frombuffer(buf, dtype=dtype, count=count * comps, offset=offset).reshape(count, comps)

    raw = np.frombuffer(buf, dtype=np.uint8, count=count * stride, offset=offset).reshape(count, stride)
    out = np.empty((count, comps), dtype=dtype)
    for i in range(count):
        out[i] = np.frombuffer(raw[i, :item_size].tobytes(), dtype=dtype, count=comps)
    return out


def resolve_material_texture_info(gltf):
    textures = gltf.get("textures", [])
    material_info = {}
    for mat_idx, material in enumerate(gltf.get("materials", [])):
        pbr = material.get("pbrMetallicRoughness", {})
        base = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        base_rgb = tuple(int(max(0, min(255, round(float(v) * 255)))) for v in base[:3])
        image_idx = -1
        tex_ref = pbr.get("baseColorTexture")
        if tex_ref is not None:
            tex_idx = tex_ref.get("index", -1)
            if 0 <= tex_idx < len(textures):
                tex_info = textures[tex_idx]
                image_idx = int(
                    tex_info.get(
                        "source",
                        tex_info.get("extensions", {}).get("GOOGLE_texture_basis", {}).get("source", -1),
                    )
                )
        material_info[mat_idx] = {
            "image_idx": image_idx,
            "base_rgb": base_rgb,
        }
    return material_info


def load_mesh_floor_triangles(mesh_dir, scan):
    mesh_path = Path(mesh_dir) / scan / f"{scan}.glb"
    if not mesh_path.exists():
        return None

    gltf, buf = load_glb(mesh_path)
    material_info = resolve_material_texture_info(gltf)
    triangles_xy = []
    triangles_mean_z = []
    triangles_normal_z_abs = []
    triangles_uv = []
    triangles_image_idx = []
    triangles_base_rgb = []
    mesh_bounds = [math.inf, -math.inf, math.inf, -math.inf]
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "POSITION" not in prim.get("attributes", {}) or "indices" not in prim:
                continue
            pos = accessor_array(gltf, buf, prim["attributes"]["POSITION"])
            idx = accessor_array(gltf, buf, prim["indices"]).reshape(-1)
            if len(idx) < 3:
                continue
            tri_indices = idx[: len(idx) // 3 * 3].reshape(-1, 3)
            tri = pos[tri_indices]
            v1 = tri[:, 1] - tri[:, 0]
            v2 = tri[:, 2] - tri[:, 0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal, axis=1) + 1e-8
            normal_z_abs = np.abs(normal[:, 2] / norm)
            tri_xy = tri[:, :, :2]
            triangles_xy.append(tri_xy)
            mean_z = tri[:, :, 2].mean(axis=1)
            triangles_mean_z.append(mean_z)
            triangles_normal_z_abs.append(normal_z_abs.astype(np.float32))

            uv_accessor = prim.get("attributes", {}).get("TEXCOORD_0")
            if uv_accessor is not None:
                tri_uv = accessor_array(gltf, buf, uv_accessor)[tri_indices].astype(np.float32)
            else:
                tri_uv = np.full((len(tri), 3, 2), np.nan, dtype=np.float32)
            triangles_uv.append(tri_uv)

            material_meta = material_info.get(prim.get("material"), {"image_idx": -1, "base_rgb": (222, 226, 232)})
            triangles_image_idx.append(np.full(len(tri), int(material_meta["image_idx"]), dtype=np.int32))
            triangles_base_rgb.append(
                np.repeat(np.asarray(material_meta["base_rgb"], dtype=np.uint8)[None, :], len(tri), axis=0)
            )

            mesh_bounds[0] = min(mesh_bounds[0], float(tri_xy[:, :, 0].min()))
            mesh_bounds[1] = max(mesh_bounds[1], float(tri_xy[:, :, 0].max()))
            mesh_bounds[2] = min(mesh_bounds[2], float(tri_xy[:, :, 1].min()))
            mesh_bounds[3] = max(mesh_bounds[3], float(tri_xy[:, :, 1].max()))

    if not triangles_xy:
        return None

    texture_arrays = {}
    try:
        texture_manifest = ensure_decoded_glb_textures(mesh_path)
        texture_arrays = load_texture_arrays(texture_manifest)
    except Exception as exc:
        print(f"[warn] texture decode failed for {scan}: {exc}")

    return {
        "triangles_xy": np.concatenate(triangles_xy, axis=0),
        "mean_z": np.concatenate(triangles_mean_z, axis=0),
        "normal_z_abs": np.concatenate(triangles_normal_z_abs, axis=0),
        "triangles_uv": np.concatenate(triangles_uv, axis=0),
        "image_idx": np.concatenate(triangles_image_idx, axis=0),
        "base_rgb": np.concatenate(triangles_base_rgb, axis=0),
        "texture_arrays": texture_arrays,
        "bounds": tuple(mesh_bounds),
        "mesh_path": str(mesh_path),
    }


def project_world_xy_to_canvas(points_xy, world_bounds, graph_box):
    xmin, ymin, xmax, ymax = graph_box
    graph_w = xmax - xmin
    graph_h = ymax - ymin
    min_x, max_x, min_y, max_y = world_bounds
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min((graph_w - 80) / span_x, (graph_h - 80) / span_y)
    pts = points_xy.copy()
    pts[:, 0] = xmin + 40 + (pts[:, 0] - min_x) * scale
    pts[:, 1] = ymax - 40 - (pts[:, 1] - min_y) * scale
    return pts


def sample_triangle_rgb(texture_arrays, image_idx, uv_tri, fallback_rgb):
    if uv_tri is None or image_idx < 0 or image_idx not in texture_arrays:
        return tuple(int(v) for v in fallback_rgb)

    tex = texture_arrays[image_idx]
    if tex.ndim != 3 or tex.shape[2] < 3:
        return tuple(int(v) for v in fallback_rgb)

    h, w = tex.shape[:2]
    sampled = []
    for uv in uv_tri:
        u = float(uv[0])
        v = float(uv[1])
        if not math.isfinite(u) or not math.isfinite(v):
            continue
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        px = int(round(u * (w - 1)))
        py = int(round((1.0 - v) * (h - 1)))
        sampled.append(tex[py, px, :3].astype(np.float32))

    if not sampled:
        return tuple(int(v) for v in fallback_rgb)
    rgb = np.mean(sampled, axis=0)
    return tuple(int(max(0, min(255, round(float(v))))) for v in rgb)


def bilinear_sample_texture(tex, uv_points):
    h, w = tex.shape[:2]
    if len(uv_points) == 0:
        return np.empty((0, 3), dtype=np.float32)

    uv = np.asarray(uv_points, dtype=np.float32)
    uv[:, 0] = np.clip(uv[:, 0], 0.0, 1.0)
    uv[:, 1] = np.clip(uv[:, 1], 0.0, 1.0)

    px = uv[:, 0] * (w - 1)
    py = (1.0 - uv[:, 1]) * (h - 1)

    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = (px - x0).astype(np.float32)
    wy = (py - y0).astype(np.float32)

    c00 = tex[y0, x0, :3].astype(np.float32)
    c10 = tex[y0, x1, :3].astype(np.float32)
    c01 = tex[y1, x0, :3].astype(np.float32)
    c11 = tex[y1, x1, :3].astype(np.float32)

    top = c00 * (1.0 - wx[:, None]) + c10 * wx[:, None]
    bot = c01 * (1.0 - wx[:, None]) + c11 * wx[:, None]
    return top * (1.0 - wy[:, None]) + bot * wy[:, None]


def rasterize_triangle_texture(rgba_canvas, tri_canvas, tex, uv_tri, alpha):
    min_x = max(int(math.floor(float(tri_canvas[:, 0].min()))), 0)
    max_x = min(int(math.ceil(float(tri_canvas[:, 0].max()))), rgba_canvas.shape[1] - 1)
    min_y = max(int(math.floor(float(tri_canvas[:, 1].min()))), 0)
    max_y = min(int(math.ceil(float(tri_canvas[:, 1].max()))), rgba_canvas.shape[0] - 1)
    if min_x > max_x or min_y > max_y:
        return

    p0, p1, p2 = tri_canvas.astype(np.float32)
    denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(float(denom)) < 1e-6:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)

    w0 = ((p1[1] - p2[1]) * (grid_x - p2[0]) + (p2[0] - p1[0]) * (grid_y - p2[1])) / denom
    w1 = ((p2[1] - p0[1]) * (grid_x - p2[0]) + (p0[0] - p2[0]) * (grid_y - p2[1])) / denom
    w2 = 1.0 - w0 - w1

    eps = 1e-4
    inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
    if not np.any(inside):
        return

    uv = (
        w0[..., None] * uv_tri[0][None, None, :]
        + w1[..., None] * uv_tri[1][None, None, :]
        + w2[..., None] * uv_tri[2][None, None, :]
    )
    sampled = bilinear_sample_texture(tex, uv[inside])
    if len(sampled) == 0:
        return

    yy, xx = np.nonzero(inside)
    yy = yy + min_y
    xx = xx + min_x
    rgba_canvas[yy, xx, :3] = np.clip(np.round(sampled), 0, 255).astype(np.uint8)
    rgba_canvas[yy, xx, 3] = alpha


def enhance_textured_patch_rgb(patch_rgb, support_mask, coverage_mask):
    local_contrast = patch_rgb.filter(ImageFilter.UnsharpMask(radius=6.0, percent=82, threshold=3))
    patch_rgb = Image.blend(patch_rgb, local_contrast, 0.52)
    patch_rgb = patch_rgb.filter(ImageFilter.UnsharpMask(radius=1.45, percent=195, threshold=2))
    patch_rgb = ImageEnhance.Contrast(patch_rgb).enhance(1.20)
    patch_rgb = ImageEnhance.Color(patch_rgb).enhance(1.05)
    patch_rgb = ImageEnhance.Brightness(patch_rgb).enhance(0.975)

    edge_shadow_mask = ImageChops.subtract(
        support_mask.filter(ImageFilter.MaxFilter(13)),
        support_mask.filter(ImageFilter.MinFilter(3)),
    ).filter(ImageFilter.GaussianBlur(radius=3.4))
    shadow_alpha = edge_shadow_mask.point(lambda v: min(92, int(v * 0.30)))
    shadow_overlay = Image.new("RGBA", patch_rgb.size, (0, 0, 0, 0))
    shadow_overlay.putalpha(shadow_alpha)
    patch_rgb = Image.alpha_composite(patch_rgb.convert("RGBA"), shadow_overlay).convert("RGB")

    # Deepen already-dark local structures so furniture, rugs, and wall contacts read more clearly.
    luminance = patch_rgb.convert("L")
    broad_light = luminance.filter(ImageFilter.GaussianBlur(radius=8.0))
    local_shadow_mask = ImageChops.subtract(broad_light, luminance).filter(ImageFilter.GaussianBlur(radius=1.8))
    local_shadow_alpha = local_shadow_mask.point(lambda v: min(76, int(v * 1.35)))
    local_shadow_overlay = Image.new("RGBA", patch_rgb.size, (10, 8, 6, 0))
    local_shadow_overlay.putalpha(local_shadow_alpha)
    patch_rgb = Image.alpha_composite(patch_rgb.convert("RGBA"), local_shadow_overlay).convert("RGB")

    highlight_mask = coverage_mask.filter(ImageFilter.GaussianBlur(radius=2.2)).point(
        lambda v: min(18, int(v * 0.045))
    )
    highlight_overlay = Image.new("RGBA", patch_rgb.size, (255, 247, 232, 0))
    highlight_overlay.putalpha(highlight_mask)
    patch_rgb = Image.alpha_composite(patch_rgb.convert("RGBA"), highlight_overlay).convert("RGB")
    return patch_rgb


def render_textured_mesh_overlay(base_image, mesh_overlay, pos3, enhanced=False):
    graph_heights = np.array([p[2] for p in pos3.values()], dtype=np.float32)
    if len(graph_heights) == 0:
        return base_image

    floor_lo = float(graph_heights.min() - 1.9)
    floor_hi = float(graph_heights.max() - 0.15)
    render_mask = (
        (mesh_overlay["mean_z"] >= floor_lo)
        & (mesh_overlay["mean_z"] <= floor_hi + 1.6)
        & (mesh_overlay["normal_z_abs"] >= 0.32)
    )
    if not np.any(render_mask):
        return base_image

    tri_xy = mesh_overlay["triangles_xy"][render_mask].copy()
    tri_uv = mesh_overlay["triangles_uv"][render_mask]
    tri_image_idx = mesh_overlay["image_idx"][render_mask]
    tri_base_rgb = mesh_overlay["base_rgb"][render_mask]
    tri_mean_z = mesh_overlay["mean_z"][render_mask]
    texture_arrays = mesh_overlay.get("texture_arrays", {})

    projected = project_world_xy_to_canvas(tri_xy.reshape(-1, 2), mesh_overlay["world_bounds"], GRAPH_BOX).reshape(-1, 3, 2)
    order = np.argsort(tri_mean_z)
    stride = max(len(order) // (520000 if enhanced else 180000), 1)
    order = order[::stride]

    gx0, gy0, gx1, gy1 = GRAPH_BOX
    graph_w = gx1 - gx0
    graph_h = gy1 - gy0
    ssaa = 3 if enhanced else 1
    render_w = graph_w * ssaa
    render_h = graph_h * ssaa

    projected_local = projected.copy()
    projected_local[:, :, 0] -= gx0
    projected_local[:, :, 1] -= gy0
    projected_local_render = projected_local * ssaa

    support_mask = Image.new("L", (render_w, render_h), 0)
    coverage_mask = Image.new("L", (render_w, render_h), 0)
    sdraw = ImageDraw.Draw(support_mask)
    cdraw = ImageDraw.Draw(coverage_mask)

    for tri, tri_local in zip(projected, projected_local_render):
        min_x = float(tri[:, 0].min())
        max_x = float(tri[:, 0].max())
        min_y = float(tri[:, 1].min())
        max_y = float(tri[:, 1].max())
        if max_x < gx0 or min_x > gx1 or max_y < gy0 or min_y > gy1:
            continue
        area = abs(
            tri[0, 0] * (tri[1, 1] - tri[2, 1])
            + tri[1, 0] * (tri[2, 1] - tri[0, 1])
            + tri[2, 0] * (tri[0, 1] - tri[1, 1])
        ) * 0.5
        if area < 0.08:
            continue
        sdraw.polygon([tuple(p) for p in tri_local], fill=255)

    textured_rgba = np.zeros((render_h, render_w, 4), dtype=np.uint8)
    solid_overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(solid_overlay, "RGBA")

    for idx in order:
        tri = projected[idx]
        tri_local = projected_local_render[idx]
        min_x = float(tri[:, 0].min())
        max_x = float(tri[:, 0].max())
        min_y = float(tri[:, 1].min())
        max_y = float(tri[:, 1].max())
        if max_x < gx0 or min_x > gx1 or max_y < gy0 or min_y > gy1:
            continue

        area = abs(
            tri[0, 0] * (tri[1, 1] - tri[2, 1])
            + tri[1, 0] * (tri[2, 1] - tri[0, 1])
            + tri[2, 0] * (tri[0, 1] - tri[1, 1])
        ) * 0.5
        if area < (0.22 if enhanced else 0.35):
            continue

        image_idx = int(tri_image_idx[idx])
        uv_ok = np.isfinite(tri_uv[idx]).all() and image_idx in texture_arrays
        alpha = 248 if tri_mean_z[idx] <= floor_hi + 0.15 else 224

        if uv_ok and area >= (1.4 if enhanced else 2.5):
            rasterize_triangle_texture(
                textured_rgba,
                tri_local,
                texture_arrays[image_idx],
                tri_uv[idx],
                alpha,
            )
            cdraw.polygon([tuple(p) for p in tri_local], fill=255)
        else:
            color = sample_triangle_rgb(texture_arrays, image_idx, tri_uv[idx], tri_base_rgb[idx])
            odraw.polygon([tuple(p) for p in tri], fill=(*color, alpha))
            cdraw.polygon([tuple(p) for p in tri_local], fill=255)

    image = base_image.convert("RGBA")
    if np.any(textured_rgba[..., 3] > 0):
        textured_overlay = Image.fromarray(textured_rgba, mode="RGBA")
        support_mask = support_mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
        support_mask = support_mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
        coverage_mask = coverage_mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))

        patch_base = Image.new("RGBA", (render_w, render_h), (246, 246, 244, 255))
        patch_rgb = Image.alpha_composite(patch_base, textured_overlay).convert("RGB")

        tiny_gap_mask = ImageChops.subtract(support_mask, coverage_mask).filter(ImageFilter.GaussianBlur(radius=0.9))
        tiny_fill = patch_rgb.filter(ImageFilter.GaussianBlur(radius=1.2))
        patch_rgb = Image.composite(tiny_fill, patch_rgb, tiny_gap_mask)

        detail_mask = coverage_mask.filter(ImageFilter.GaussianBlur(radius=0.8))
        sharpened_detail = patch_rgb.filter(ImageFilter.UnsharpMask(radius=0.9, percent=105, threshold=2))
        patch_rgb = Image.composite(sharpened_detail, patch_rgb, detail_mask)

        if enhanced:
            patch_rgb = enhance_textured_patch_rgb(patch_rgb, support_mask, coverage_mask)
            patch_rgb = patch_rgb.resize((graph_w, graph_h), RESAMPLING.LANCZOS)
            support_mask = support_mask.resize((graph_w, graph_h), RESAMPLING.LANCZOS)
        else:
            patch_rgb = ImageEnhance.Contrast(patch_rgb).enhance(1.06)
            patch_rgb = ImageEnhance.Color(patch_rgb).enhance(1.02)

        textured_overlay = patch_rgb.convert("RGBA")
        textured_overlay.putalpha(support_mask.point(lambda v: min(250, v)))
        image.alpha_composite(textured_overlay, dest=(gx0, gy0))

    image = Image.alpha_composite(image, solid_overlay)

    graph_patch = image.crop(GRAPH_BOX)
    graph_patch = ImageEnhance.Contrast(graph_patch).enhance(1.04 if enhanced else 1.02)
    graph_patch = ImageEnhance.Color(graph_patch).enhance(1.015 if enhanced else 1.01)
    image.paste(graph_patch, GRAPH_BOX)
    return image.convert("RGB")


def render_occupancy_mesh_overlay(base_image, mesh_overlay, pos3):
    image = base_image
    graph_heights = np.array([p[2] for p in pos3.values()], dtype=np.float32)
    if len(graph_heights) == 0:
        return image

    floor_lo = float(graph_heights.min() - 1.9)
    floor_hi = float(graph_heights.max() - 0.15)
    floor_mask = (mesh_overlay["mean_z"] >= floor_lo) & (mesh_overlay["mean_z"] <= floor_hi)
    floor_triangles = mesh_overlay["triangles_xy"][floor_mask]
    if len(floor_triangles) == 0:
        return image

    projected = project_world_xy_to_canvas(
        floor_triangles.reshape(-1, 2), mesh_overlay["world_bounds"], GRAPH_BOX
    ).reshape(-1, 3, 2)
    graph_w = GRAPH_BOX[2] - GRAPH_BOX[0]
    graph_h = GRAPH_BOX[3] - GRAPH_BOX[1]
    overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    local_mask = Image.new("L", (graph_w, graph_h), 0)
    mdraw = ImageDraw.Draw(local_mask)
    centroids = projected.mean(axis=1)
    keep = (
        (centroids[:, 0] >= GRAPH_BOX[0] - 20)
        & (centroids[:, 0] <= GRAPH_BOX[2] + 20)
        & (centroids[:, 1] >= GRAPH_BOX[1] - 20)
        & (centroids[:, 1] <= GRAPH_BOX[3] + 20)
    )
    centroids = centroids[keep]
    stride = max(len(centroids) // 18000, 1)
    dot_radius = 3
    for c in centroids[::stride]:
        cx = float(c[0] - GRAPH_BOX[0])
        cy = float(c[1] - GRAPH_BOX[1])
        mdraw.ellipse((cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius), fill=255)

    local_mask = local_mask.filter(ImageFilter.MaxFilter(17))
    local_mask = local_mask.filter(ImageFilter.MaxFilter(17))
    local_mask = local_mask.filter(ImageFilter.GaussianBlur(8))
    alpha_mask = local_mask.point(lambda v: 165 if v >= 6 else 0)
    occ_rgba = Image.new("RGBA", (graph_w, graph_h), (178, 205, 226, 0))
    occ_rgba.putalpha(alpha_mask)
    overlay.paste(occ_rgba, (GRAPH_BOX[0], GRAPH_BOX[1]), occ_rgba)

    bbox = alpha_mask.getbbox()
    if bbox is not None:
        bx0, by0, bx1, by1 = bbox
        odraw.rounded_rectangle(
            (GRAPH_BOX[0] + bx0, GRAPH_BOX[1] + by0, GRAPH_BOX[0] + bx1, GRAPH_BOX[1] + by1),
            radius=12,
            outline=(132, 160, 186, 255),
            width=2,
        )

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def load_true3d_bev(scan, true3d_root):
    if not true3d_root:
        return None
    scan_dir = Path(true3d_root) / scan
    bev_path = scan_dir / "rgb_bev.png"
    meta_path = scan_dir / "meta.json"
    if not bev_path.exists() or not meta_path.exists():
        return None
    meta = load_json(meta_path)
    world_bounds = tuple(meta.get("world_bounds", []))
    if len(world_bounds) != 4:
        return None
    return {
        "image_path": str(bev_path),
        "world_bounds": world_bounds,
        "image_size": meta.get("image_size"),
    }


def render_true3d_bev_background(base_image, bev_info):
    gx0, gy0, gx1, gy1 = GRAPH_BOX
    graph_w = gx1 - gx0
    graph_h = gy1 - gy0
    min_x, max_x, min_y, max_y = bev_info["world_bounds"]
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min((graph_w - 80) / span_x, (graph_h - 80) / span_y)
    target_w = max(int(round(span_x * scale)), 1)
    target_h = max(int(round(span_y * scale)), 1)

    patch = Image.open(bev_info["image_path"]).convert("RGB")
    patch = patch.resize((target_w, target_h), getattr(Image, "Resampling", Image).LANCZOS)
    out = base_image.copy()
    out.paste(patch, (gx0 + 40, gy1 - 40 - target_h))
    return out


def render_static_background(scan, graph, disp_pos, pos3, gt_path, mesh_overlay=None, mesh_enhanced=False, true3d_bev=None):
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), (248, 249, 251))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(GRAPH_BOX, radius=20, fill=(255, 255, 255), outline=(210, 214, 220), width=2)
    draw.rounded_rectangle(SIDEBAR_BOX, radius=20, fill=(255, 255, 255), outline=(210, 214, 220), width=2)

    if true3d_bev is not None:
        image = render_true3d_bev_background(image, true3d_bev)
        draw = ImageDraw.Draw(image)
    elif mesh_overlay is not None:
        if mesh_overlay.get("texture_arrays"):
            image = render_textured_mesh_overlay(image, mesh_overlay, pos3, enhanced=mesh_enhanced)
        else:
            image = render_occupancy_mesh_overlay(image, mesh_overlay, pos3)
        draw = ImageDraw.Draw(image)

    for a, b in graph.edges():
        pa = disp_pos[a]
        pb = disp_pos[b]
        draw.line((GRAPH_BOX[0] + pa[0], GRAPH_BOX[1] + pa[1], GRAPH_BOX[0] + pb[0], GRAPH_BOX[1] + pb[1]), fill=(225, 228, 232), width=2)

    for vp, (px, py) in disp_pos.items():
        r = 4
        draw.ellipse((GRAPH_BOX[0] + px - r, GRAPH_BOX[1] + py - r, GRAPH_BOX[0] + px + r, GRAPH_BOX[1] + py + r), fill=(185, 190, 198))

    gt_pts = [(GRAPH_BOX[0] + disp_pos[vp][0], GRAPH_BOX[1] + disp_pos[vp][1]) for vp in gt_path if vp in disp_pos]
    draw_polyline(draw, gt_pts, (80, 170, 90), 6)
    start_vp = gt_path[0]
    goal_vp = gt_path[-1]
    if start_vp in disp_pos:
        px, py = disp_pos[start_vp]
        draw.ellipse((GRAPH_BOX[0] + px - 10, GRAPH_BOX[1] + py - 10, GRAPH_BOX[0] + px + 10, GRAPH_BOX[1] + py + 10), fill=(66, 133, 244))
    if goal_vp in disp_pos:
        px, py = disp_pos[goal_vp]
        draw_star(draw, (GRAPH_BOX[0] + px, GRAPH_BOX[1] + py), 14, (52, 168, 83))

    title_font = load_font(28)
    small_font = load_font(16)
    draw.text((GRAPH_BOX[0] + 20, GRAPH_BOX[1] + 14), "Mesh Top-Down + Topological Graph", font=title_font, fill=(31, 41, 55))
    if true3d_bev is not None:
        overlay_label = "true 3D orthographic mesh render"
    elif mesh_overlay is None:
        overlay_label = "graph-only fallback"
    elif mesh_overlay.get("texture_arrays"):
        overlay_label = "enhanced textured mesh overlay" if mesh_enhanced else "textured mesh overlay"
    else:
        overlay_label = "occupancy mesh fallback"
    subtitle = f"scan: {scan} | topo graph projected onto mesh x-y plane | {overlay_label}"
    draw.text((GRAPH_BOX[0] + 20, GRAPH_BOX[1] + 54), subtitle, font=small_font, fill=(107, 114, 128))
    return image


def draw_polyline(draw, pts, color, width):
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width, joint="curve")


def draw_star(draw, center, radius, color):
    cx, cy = center
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    draw.polygon(pts, fill=color)


def load_font(size):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def expand_r2r_annotations(anno_path):
    items = load_json(anno_path)
    out = {}
    for item in items:
        pid = item["path_id"]
        for idx, instr in enumerate(item["instructions"]):
            out[f"{pid}_{idx}"] = {
                "scan": item["scan"],
                "instruction": instr,
                "gt_path": item["path"],
                "heading": item.get("heading"),
            }
    return out


def expand_reverie_annotations(anno_path):
    items = load_json(anno_path)
    out = {}
    for item in items:
        path_id = item["path_id"]
        obj_id = str(item["objId"])
        for idx, instr in enumerate(item["instructions"]):
            out[f"{path_id}_{obj_id}_{idx}"] = {
                "scan": item["scan"],
                "instruction": instr,
                "gt_path": item["path"],
                "obj_id": obj_id,
                "heading": item.get("heading"),
                "raw_id": item.get("id"),
            }
    return out


def draw_text_block(draw, xy, title, lines, title_font, body_font, fill=(20, 20, 20), line_gap=8):
    x, y = xy
    draw.text((x, y), title, font=title_font, fill=fill)
    y += title_font.size + 8
    for line in lines:
        draw.text((x, y), line, font=body_font, fill=fill)
        y += body_font.size + line_gap
    return y


def render_episode_frames(dataset, pred_item, meta, graph, disp_pos, pos3, bbox_meta=None, mesh_overlay=None, mesh_enhanced=False, true3d_bev=None):
    pred_path = flatten_pred_trajectory(pred_item["trajectory"])
    gt_path = meta["gt_path"]
    scan = meta["scan"]
    instruction = meta["instruction"]
    obj_id = meta.get("obj_id")
    pred_objid = pred_item.get("pred_objid")
    start_vp = gt_path[0]
    goal_vp = gt_path[-1]

    title_font = load_font(28)
    body_font = load_font(20)
    small_font = load_font(16)

    frames = []
    xmin, ymin, xmax, ymax = GRAPH_BOX
    graph_w = xmax - xmin
    graph_h = ymax - ymin

    static_bg = render_static_background(
        scan,
        graph,
        disp_pos,
        pos3,
        gt_path,
        mesh_overlay=mesh_overlay,
        mesh_enhanced=mesh_enhanced,
        true3d_bev=true3d_bev,
    )

    for step_idx, current_vp in enumerate(pred_path):
        image = static_bg.copy()
        draw = ImageDraw.Draw(image)

        # predicted path so far
        pred_so_far = pred_path[: step_idx + 1]
        pred_pts = [(xmin + disp_pos[vp][0], ymin + disp_pos[vp][1]) for vp in pred_so_far if vp in disp_pos]
        draw_polyline(draw, pred_pts, (219, 68, 55), 7)

        # current viewpoint
        if current_vp in disp_pos:
            px, py = disp_pos[current_vp]
            draw.ellipse((xmin + px - 12, ymin + py - 12, xmin + px + 12, ymin + py + 12), fill=(244, 180, 0), outline=(160, 100, 0), width=2)

        # neighbors / candidates
        if current_vp in graph:
            for nb in graph.neighbors(current_vp):
                if nb not in disp_pos:
                    continue
                px, py = disp_pos[nb]
                draw.ellipse((xmin + px - 8, ymin + py - 8, xmin + px + 8, ymin + py + 8), outline=(255, 140, 0), width=2)

        # next chosen step
        next_vp = pred_path[step_idx + 1] if step_idx + 1 < len(pred_path) else None
        if next_vp and current_vp in disp_pos and next_vp in disp_pos:
            px1, py1 = disp_pos[current_vp]
            px2, py2 = disp_pos[next_vp]
            draw.line((xmin + px1, ymin + py1, xmin + px2, ymin + py2), fill=(156, 39, 176), width=5)

        # Sidebar info
        sx, sy, _, _ = SIDEBAR_BOX
        y = sy + 20
        info_lines = [
            f"dataset: {dataset}",
            f"instr_id: {pred_item['instr_id']}",
            f"scan: {scan}",
            f"step: {step_idx + 1}/{len(pred_path)}",
            f"current_vp: {current_vp}",
            f"next_vp: {next_vp or 'STOP'}",
            f"start_vp: {start_vp}",
            f"goal_vp: {goal_vp}",
        ]
        if obj_id is not None:
            visible_here = "unknown"
            if bbox_meta is not None:
                key = f"{scan}_{current_vp}"
                visible_here = "yes" if key in bbox_meta and obj_id in bbox_meta[key] else "no"
            info_lines.extend([
                f"target_objid: {obj_id}",
                f"pred_objid: {pred_objid}",
                f"target_visible_here: {visible_here}",
            ])
        y = draw_text_block(draw, (sx + 20, y), "Episode", info_lines, title_font, body_font)

        wrapped = textwrap.wrap(instruction, width=38)
        y += 18
        y = draw_text_block(draw, (sx + 20, y), "Instruction", wrapped, title_font, body_font)

        legend = [
            "textured mesh: bird's-eye scan overlay",
            "blue: start",
            "green star: goal / GT target",
            "green line: GT path",
            "red line: predicted path so far",
            "orange node: current viewpoint",
            "orange rings: navigable neighbors",
            "purple line: next chosen move",
        ]
        y += 18
        draw_text_block(draw, (sx + 20, y), "Legend", legend, title_font, body_font)

        frames.append(image)
    return frames


def save_frames_and_movies(frames, episode_dir, fps=2):
    ensure_dir(episode_dir)
    frame_dir = os.path.join(episode_dir, "frames")
    ensure_dir(frame_dir)

    for idx, frame in enumerate(frames):
        frame.save(os.path.join(frame_dir, f"frame_{idx:03d}.png"))

    gif_path = os.path.join(episode_dir, "trajectory.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / max(fps, 1)),
        loop=0,
    )

    mp4_path = os.path.join(episode_dir, "trajectory.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(frame_dir, "frame_%03d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        mp4_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return gif_path, mp4_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["r2r", "reverie"], required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--connectivity_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bbox_json", default=None)
    parser.add_argument("--mesh_dir", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--instr_id", default=None)
    parser.add_argument("--mesh_enhanced", action="store_true")
    parser.add_argument("--true3d_root", default=None)
    args = parser.parse_args()

    preds = load_json(args.preds)
    if args.dataset == "r2r":
        anno_lookup = expand_r2r_annotations(args.annotations)
    else:
        anno_lookup = expand_reverie_annotations(args.annotations)
    bbox_meta = load_json(args.bbox_json) if args.bbox_json else None

    preds = [p for p in preds if p.get("instr_id") in anno_lookup]
    if args.instr_id:
        preds = [p for p in preds if p.get("instr_id") == args.instr_id]
        if not preds:
            raise KeyError(f"instr_id {args.instr_id} not found in predictions after annotation filtering")
    if args.seed is not None:
        rng = random.Random(args.seed)
        rng.shuffle(preds)
        print(f"[info] sampling {min(args.limit, len(preds))} episode(s) with seed={args.seed}")

    ensure_dir(args.output_dir)
    graph_cache = {}
    mesh_cache = {}
    manifest = []

    for pred_item in preds[: args.limit]:
        instr_id = pred_item["instr_id"]
        meta = anno_lookup[instr_id]
        scan = meta["scan"]
        if scan not in graph_cache:
            graph, pos3 = load_graph_for_scan(args.connectivity_dir, scan)
            graph_cache[scan] = (graph, pos3)
        graph, pos3 = graph_cache[scan]

        mesh_overlay = None
        world_bounds = None
        true3d_bev = load_true3d_bev(scan, args.true3d_root)
        if true3d_bev is not None:
            world_bounds = tuple(true3d_bev["world_bounds"])
        elif args.mesh_dir:
            if scan not in mesh_cache:
                mesh_cache[scan] = load_mesh_floor_triangles(args.mesh_dir, scan)
            mesh_overlay = mesh_cache[scan]
            if mesh_overlay is not None:
                world_bounds = pad_world_bounds(mesh_overlay["bounds"])
                mesh_overlay = dict(mesh_overlay)
                mesh_overlay["world_bounds"] = world_bounds

        disp_pos, _ = build_display_positions(
            pos3,
            GRAPH_BOX[2] - GRAPH_BOX[0],
            GRAPH_BOX[3] - GRAPH_BOX[1],
            bounds=world_bounds,
        )
        frames = render_episode_frames(
            args.dataset,
            pred_item,
            meta,
            graph,
            disp_pos,
            pos3,
            bbox_meta=bbox_meta,
            mesh_overlay=mesh_overlay,
            mesh_enhanced=args.mesh_enhanced,
            true3d_bev=true3d_bev,
        )
        safe_id = instr_id.replace("/", "_")
        episode_dir = os.path.join(args.output_dir, safe_id)
        gif_path, mp4_path = save_frames_and_movies(frames, episode_dir, fps=args.fps)
        manifest.append({
            "instr_id": instr_id,
            "scan": scan,
            "frames": len(frames),
            "gif": gif_path,
            "mp4": mp4_path,
        })
        print(f"[done] {instr_id} -> {episode_dir}")

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"wrote manifest: {os.path.join(args.output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
