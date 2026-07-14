"""Self-contained SOMA-30 skeleton renderer, kimodo ``render_hml3d`` style: joints [T,30,3] -> mp4.

Pure matplotlib + imageio (pyav h264, imageio-ffmpeg fallback), NO kimodo dependency, so it runs
in the SAME (cosmos) env as training and can be called directly from train.py's do_viz and
eval_all.py.

VISUAL STYLE (a faithful adaptation of ``kimodo/scripts/render_hml3d.py`` + the parents-based
``render_soma.py`` skeleton drawer, with the SOMA-30 kinematic chain instead of HML3D-22):
  * a fixed-size square XZ viewport (``DEFAULT_VIEW_HALF`` = 2 m half-extent) TRACKS each
    skeleton's root per frame, so the character stays a consistent on-screen size;
  * a gray floor quad + 1 m world-coordinate grid (``DEFAULT_GRID_SPACING``) spans the whole
    motion -- grid lines visibly scroll past the tracking camera so translation reads cleanly;
  * the root XZ trajectory is drawn on the floor with a marker at the current frame;
  * side-by-side layout: GT (left, blue) | generated (right, red); ``gt_joints=None`` keeps the
    blank left panel so layout stays consistent (title "(no GT)");
  * fingertip-end joints (LeftHandThumbEnd/LeftHandMiddleEnd/Right...) are skipped, matching
    kimodo's SOMA viz.

COORDINATES: input joints are world Y-up / +Z-forward (the uniego decode convention -- same as
kimodo). In this right-handed basis +X points to the character's LEFT (SOMA-30 T-pose:
LeftFoot.x = +0.10, RightFoot.x = -0.10), and matplotlib renders +X on screen RIGHT, so a literal
mapping mirrors the scene (a right turn reads as a left turn). We therefore negate world X at the
entry of every render function ("display frame", exactly kimodo's ``_to_display``), then remap
display (x, y, z) -> mplot3d's Z-up plot (x, z, y). This SUPERSEDES the previous renderer here,
which did the Z-up swap only (no X negation, no floor, per-clip auto-zoom).

Entry points:
  * ``render_sidebyside(gt_joints, gen_joints, ...)``  -> (T, H, W, 3) uint8 frames
  * ``render_single(joints, ...)``                     -> (T, H, W, 3) uint8 frames
  * ``write_mp4(frames, path, fps)``                   -> h264 mp4 (pyav; imageio-ffmpeg fallback)
  * ``render_motion_mp4(joints, out_path, caption, fps, gt_joints=None)`` -- the train.py /
    eval_all.py call seam: single-panel when ``gt_joints`` is None, GT|gen side-by-side otherwise.
  * ``render_conditioned_motion_mp4(...)``             -- conditioning image | GT | generated.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

# SOMA-30 bone parents (skeleton_soma30.npz['parents']); -1 == root (Hips).
PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19,
           0, 22, 23, 24, 0, 26, 27, 28]
# Fingertip-end joints kimodo drops from the SOMA stick figure (skeleton_soma30 names
# LeftHandThumbEnd/LeftHandMiddleEnd/RightHandThumbEnd/RightHandMiddleEnd).
SKIP_JOINTS = (14, 15, 20, 21)

FLOOR_COLOR = "#cfcfcf"
FLOOR_EDGE = "#8a8a8a"
GRID_COLOR = "#a0a0a0"

DEFAULT_VIEW_HALF = 2.0   # half-extent of the (square) XZ viewport, in meters
DEFAULT_GRID_SPACING = 1.0


# -----------------------------------------------------------------------------
# Geometry helpers (kimodo render_hml3d verbatim)
# -----------------------------------------------------------------------------
def _to_display(joints: np.ndarray) -> np.ndarray:
    """World -> display frame. Negates X so character-right ends up on screen right."""
    out = np.asarray(joints, dtype=np.float32).copy()
    out[..., 0] = -out[..., 0]
    return out


def _world_extent(
    joints_list: List[np.ndarray],
    extra_pts: Optional[List[np.ndarray]] = None,
    xz_pad: float = 1.0,
    y_top_pad: float = 0.2,
    min_y_top: float = 2.0,
) -> dict:
    """Full motion bounding box (used to size the floor + grid)."""
    all_pts = [j.reshape(-1, 3) for j in joints_list if j is not None]
    if extra_pts:
        all_pts.extend(p.reshape(-1, 3) for p in extra_pts if p is not None)
    arr = np.concatenate(all_pts, axis=0)
    return {
        "x": (float(arr[:, 0].min()) - xz_pad, float(arr[:, 0].max()) + xz_pad),
        "z": (float(arr[:, 2].min()) - xz_pad, float(arr[:, 2].max()) + xz_pad),
        "y_top": max(min_y_top, float(arr[:, 1].max()) + y_top_pad),
    }


def _viewport_for_center(center_xz: Tuple[float, float], view_half: float, y_top: float) -> dict:
    """Square XZ viewport centered at ``center_xz`` with Y running 0 -> ``y_top``."""
    cx, cz = float(center_xz[0]), float(center_xz[1])
    return {
        "x": (cx - view_half, cx + view_half),
        "z": (cz - view_half, cz + view_half),
        "y_top": float(y_top),
    }


def _draw_floor_and_grid(ax, extent: dict, spacing: float = DEFAULT_GRID_SPACING,
                         floor_alpha: float = 0.55):
    """Gray floor quad + world-coordinate grid lines spanning the full motion extent."""
    (xmin, xmax) = extent["x"]
    (zmin, zmax) = extent["z"]
    quad_world = np.array(
        [[xmin, 0.0, zmin], [xmax, 0.0, zmin], [xmax, 0.0, zmax], [xmin, 0.0, zmax]],
        dtype=np.float32,
    )
    quad_plot = quad_world[:, [0, 2, 1]]  # world (x,y,z) -> plot (x,z,y)
    ax.add_collection3d(
        Poly3DCollection([quad_plot], facecolors=FLOOR_COLOR, edgecolors="none",
                         alpha=floor_alpha, zorder=0)
    )
    segs: List[List[List[float]]] = []
    z0 = np.floor(zmin / spacing) * spacing
    z1 = np.ceil(zmax / spacing) * spacing
    for z in np.arange(z0, z1 + 0.5 * spacing, spacing):
        if zmin <= z <= zmax:
            segs.append([[xmin, float(z), 0.0], [xmax, float(z), 0.0]])
    x0 = np.floor(xmin / spacing) * spacing
    x1 = np.ceil(xmax / spacing) * spacing
    for x in np.arange(x0, x1 + 0.5 * spacing, spacing):
        if xmin <= x <= xmax:
            segs.append([[float(x), zmin, 0.0], [float(x), zmax, 0.0]])
    if segs:
        ax.add_collection3d(
            Line3DCollection(segs, colors=GRID_COLOR, linewidths=0.7, alpha=0.85, zorder=1)
        )


def _draw_trajectory(ax, root_world: np.ndarray, current_t: int, color: str,
                     lw: float = 1.8, marker_size: float = 40.0):
    """Root XZ trajectory projected onto the floor (y=0), with a marker at frame t."""
    if root_world is None or len(root_world) == 0:
        return
    trail = root_world.copy()
    trail[:, 1] = 0.0
    ax.plot(trail[:, 0], trail[:, 2], trail[:, 1],
            color=color, linewidth=lw, alpha=0.9, zorder=2)
    ti = min(int(current_t), len(trail) - 1)
    ax.scatter(trail[ti, 0], trail[ti, 2], trail[ti, 1],
               c=color, s=marker_size, edgecolors="white", linewidths=1.0, zorder=6)


def _draw_skel(ax, joints_xyz: np.ndarray, joint_parents: Sequence[int], color: str,
               lw: float = 2.2, scatter_s: float = 18.0,
               skip_joints: Optional[Sequence[int]] = SKIP_JOINTS) -> None:
    """Stick figure: one segment per (parent, child) pair (parents-based, render_soma style)."""
    skip = {int(j) for j in skip_joints} if skip_joints else set()
    keep = [j for j in range(joints_xyz.shape[0]) if j not in skip]
    ax.scatter(joints_xyz[keep, 0], joints_xyz[keep, 2], joints_xyz[keep, 1],
               c=color, s=scatter_s, zorder=5)
    for child, parent in enumerate(joint_parents):
        p = int(parent)
        if p < 0 or child in skip or p in skip:
            continue
        ax.plot([joints_xyz[p, 0], joints_xyz[child, 0]],
                [joints_xyz[p, 2], joints_xyz[child, 2]],
                [joints_xyz[p, 1], joints_xyz[child, 1]],
                color=color, linewidth=lw, zorder=4)


def _setup_axes(ax, viewport: dict, title: str, view_elev: float = 20.0, view_azim: float = -60.0):
    (xmin, xmax) = viewport["x"]
    (zmin, zmax) = viewport["z"]
    ymax = viewport["y_top"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(0.0, ymax)
    try:
        ax.set_box_aspect([xmax - xmin, zmax - zmin, ymax])
    except Exception:  # noqa: BLE001
        pass
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("z (forward)", fontsize=7)
    ax.set_zlabel("y (up)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9)
    ax.view_init(elev=view_elev, azim=view_azim)


def _frame_indices(T: int, max_frames: Optional[int], frame_stride: int) -> List[int]:
    idx = list(range(0, T, max(1, frame_stride)))
    if max_frames is not None and len(idx) > max_frames:
        idx = list(np.linspace(0, T - 1, max_frames, dtype=int))
    return idx


# -----------------------------------------------------------------------------
# Top-level renderers
# -----------------------------------------------------------------------------
def render_sidebyside(
    gt_joints: Optional[np.ndarray],  # (T, J, 3) or None for "no GT" (blank left panel)
    gen_joints: np.ndarray,
    joint_parents: Sequence[int] = PARENTS,
    caption: str = "",
    width: int = 1200,
    height: int = 600,
    dpi: int = 120,
    max_frames: Optional[int] = 200,
    frame_stride: int = 1,
    view_half: float = DEFAULT_VIEW_HALF,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    skip_joints: Optional[Sequence[int]] = SKIP_JOINTS,
) -> np.ndarray:
    """GT (left, blue) vs generated (right, red), per-skeleton tracking camera.

    Returns ``(T_render, H, W, 3)`` uint8 frames.
    """
    gen_joints = _to_display(gen_joints)
    if gt_joints is not None:
        gt_joints = _to_display(gt_joints)
    T = gen_joints.shape[0] if gt_joints is None else min(gt_joints.shape[0], gen_joints.shape[0])
    idx = _frame_indices(T, max_frames, frame_stride)
    gen_trail = gen_joints[:T, 0]
    if gt_joints is None:
        gt_trail = gen_trail  # empty-GT panel still tracks something sensible
        extent = _world_extent([gen_joints[:T]], extra_pts=[gen_trail])
    else:
        gt_trail = gt_joints[:T, 0]
        extent = _world_extent([gt_joints[:T], gen_joints[:T]], extra_pts=[gt_trail, gen_trail])

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    frames: List[np.ndarray] = []
    try:
        for t in idx:
            fig.clf()
            gt_root = gt_trail[t]
            gen_root = gen_trail[t]

            ax_gt = fig.add_subplot(1, 2, 1, projection="3d")
            _setup_axes(
                ax_gt,
                _viewport_for_center((gt_root[0], gt_root[2]), view_half, extent["y_top"]),
                (f"GT  t={t}" if gt_joints is not None else "(no GT)"),
            )
            _draw_floor_and_grid(ax_gt, extent, spacing=grid_spacing)
            if gt_joints is not None:
                _draw_trajectory(ax_gt, gt_trail, t, "tab:blue")
                _draw_skel(ax_gt, gt_joints[t], joint_parents, "tab:blue",
                           skip_joints=skip_joints)

            ax_gen = fig.add_subplot(1, 2, 2, projection="3d")
            _setup_axes(
                ax_gen,
                _viewport_for_center((gen_root[0], gen_root[2]), view_half, extent["y_top"]),
                f"generated  t={t}",
            )
            _draw_floor_and_grid(ax_gen, extent, spacing=grid_spacing)
            _draw_trajectory(ax_gen, gen_trail, t, "tab:red")
            _draw_skel(ax_gen, gen_joints[t], joint_parents, "tab:red", skip_joints=skip_joints)

            if caption:
                fig.suptitle(caption[:100], fontsize=10)
            fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.04, wspace=0.08)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(img)
    finally:
        plt.close(fig)
    return np.stack(frames, axis=0)


def render_single(
    joints: np.ndarray,
    joint_parents: Sequence[int] = PARENTS,
    caption: str = "",
    color: str = "tab:red",
    width: int = 700,
    height: int = 700,
    dpi: int = 120,
    max_frames: Optional[int] = 200,
    frame_stride: int = 1,
    view_half: float = DEFAULT_VIEW_HALF,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    skip_joints: Optional[Sequence[int]] = SKIP_JOINTS,
) -> np.ndarray:
    """One skeleton + its trail, tracking camera + world-grid floor. -> (T,H,W,3) uint8."""
    joints = _to_display(joints)
    T = joints.shape[0]
    idx = _frame_indices(T, max_frames, frame_stride)
    trail = joints[:T, 0]
    extent = _world_extent([joints[:T]], extra_pts=[trail])

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    frames: List[np.ndarray] = []
    try:
        for t in idx:
            fig.clf()
            root = trail[t]
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            _setup_axes(
                ax,
                _viewport_for_center((root[0], root[2]), view_half, extent["y_top"]),
                f"t={t}",
            )
            _draw_floor_and_grid(ax, extent, spacing=grid_spacing)
            _draw_trajectory(ax, trail, t, color)
            _draw_skel(ax, joints[t], joint_parents, color, skip_joints=skip_joints)
            if caption:
                fig.suptitle(caption[:100], fontsize=10)
            fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.04)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames.append(img)
    finally:
        plt.close(fig)
    return np.stack(frames, axis=0)


# -----------------------------------------------------------------------------
# MP4 writer: pyav h264 (kimodo's writer) with imageio-ffmpeg libx264 fallback
# -----------------------------------------------------------------------------
def write_mp4(frames: np.ndarray, out_path: str, fps: int = 20) -> str:
    """(T,H,W,3) uint8 -> h264 mp4. Prefers the pyav plugin (kimodo's exact writer);
    falls back to the legacy imageio ffmpeg writer when pyav is unavailable."""
    try:
        import imageio.v3 as iio
        iio.imwrite(str(out_path), frames, fps=float(fps), codec="h264", plugin="pyav")
    except Exception:  # noqa: BLE001 -- pyav missing/failed: legacy imageio-ffmpeg writer
        import imageio.v2 as imageio
        imageio.mimsave(out_path, list(frames), fps=fps, codec="libx264", quality=8,
                        macro_block_size=None)
    return out_path


def render_motion_mp4(joints, out_path: str, caption: str = "", fps: int = 20,
                      parents=PARENTS, gt_joints=None, **kw) -> str:
    """The train.py / eval_all.py seam: joints [T,30,3] (Y-up, +Z-fwd) -> skeleton mp4.

    ``gt_joints=None`` -> single tracking-camera panel (red); with ``gt_joints`` [T,30,3]
    -> kimodo-style GT (left, blue) | generated (right, red) side-by-side.
    """
    if gt_joints is None:
        frames = render_single(np.asarray(joints, dtype=np.float32),
                               joint_parents=parents, caption=caption, **kw)
    else:
        frames = render_sidebyside(np.asarray(gt_joints, dtype=np.float32),
                                   np.asarray(joints, dtype=np.float32),
                                   joint_parents=parents, caption=caption, **kw)
    return write_mp4(frames, out_path, fps=fps)


def _image_to_uint8_hwc(image) -> np.ndarray:
    """Accept torch/numpy CHW or HWC image data and return contiguous RGB uint8 HWC."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"condition_image must have 3 dimensions, got {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"condition_image must be RGB CHW/HWC, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def render_conditioned_motion_mp4(
    *,
    condition_image,
    gen_joints,
    out_path: str,
    gt_joints=None,
    condition_out_path: Optional[str] = None,
    caption: str = "",
    fps: int = 20,
    parents=PARENTS,
    **kw,
) -> str:
    """Render TI2M as conditioning image | GT motion | generated motion.

    The source image is saved separately when ``condition_out_path`` is provided. The MP4 repeats
    a letterboxed image panel beside the normal two-panel motion rendering, making every training
    visualization self-contained without altering the motion renderer's camera or scale.
    """
    from PIL import Image, ImageDraw

    condition = _image_to_uint8_hwc(condition_image)
    condition_pil = Image.fromarray(condition, mode="RGB")
    if condition_out_path is not None:
        condition_pil.save(condition_out_path)

    motion_frames = render_sidebyside(
        None if gt_joints is None else np.asarray(gt_joints, dtype=np.float32),
        np.asarray(gen_joints, dtype=np.float32),
        joint_parents=parents,
        caption=caption,
        **kw,
    )
    height = int(motion_frames.shape[1])
    panel_width = height
    margin = max(12, height // 40)
    label_height = max(28, height // 16)
    available_w = panel_width - 2 * margin
    available_h = height - 2 * margin - label_height
    scale = min(available_w / condition_pil.width, available_h / condition_pil.height)
    resized = condition_pil.resize(
        (
            max(1, int(round(condition_pil.width * scale))),
            max(1, int(round(condition_pil.height * scale))),
        ),
        resample=Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", (panel_width, height), color=(245, 245, 245))
    x = (panel_width - resized.width) // 2
    y = margin + label_height + max(0, (available_h - resized.height) // 2)
    panel.paste(resized, (x, y))
    ImageDraw.Draw(panel).text(
        (margin, margin), "conditioning image", fill=(25, 25, 25)
    )
    panel_np = np.asarray(panel, dtype=np.uint8)
    panels = np.broadcast_to(panel_np, (motion_frames.shape[0], *panel_np.shape)).copy()
    frames = np.concatenate([panels, motion_frames], axis=2)
    return write_mp4(frames, out_path, fps=fps)


if __name__ == "__main__":
    import sys, json, os
    vdir = sys.argv[1]
    for it in json.load(open(os.path.join(vdir, "manifest.json"))):
        j = np.load(it["joints_npy"])
        out = it["joints_npy"].replace(".npy", ".mp4")
        gt = None
        gtp = it.get("gt_joints_npy")
        if gtp and os.path.isfile(gtp):
            gt = np.load(gtp)
        render_motion_mp4(j, out, caption=it.get("caption", ""), gt_joints=gt)
        print("rendered", os.path.basename(out), j.shape)
