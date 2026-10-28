#!/usr/bin/env python3
"""
Render a normalized PLY using synthetic orbit camera poses.
No COLMAP needed. Load PLY -> sample orbit poses -> render -> save images.

Usage:
  python render_synthetic.py -i normalized.ply -o render_output/ --num_views 20

  # With model_path (required for RendererAdapter, use any valid path):
  python render_synthetic.py -i normalized.ply -o render_output/ -m output/exp/xyz --num_views 20
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# scene / utils / arguments / gaussian_renderer come from the repo root (shared with training)
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from argparse import Namespace
import torch.nn.functional as F


# ----------------------------
# ROI crop (OVSeg-style) for CLIP-ready output
# ----------------------------
def _expand_box(x1: float, y1: float, x2: float, y2: float, expand_ratio: float = 1.0, max_h: int = None, max_w: int = None):
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    w, h = (x2 - x1) * expand_ratio, (y2 - y1) * expand_ratio
    box = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    if max_h: box[1], box[3] = max(0, box[1]), min(max_h - 1, box[3])
    if max_w: box[0], box[2] = max(0, box[0]), min(max_w - 1, box[2])
    return [int(b) for b in box]


def _mask_to_bbox(mask: torch.Tensor):
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    row = torch.nonzero(mask.sum(dim=0))[:, 0]
    if len(row) == 0:
        return None
    x1, x2 = int(row.min().item()), int(row.max().item()) + 1
    col = torch.nonzero(mask.sum(dim=1))[:, 0]
    if len(col) == 0:
        return None
    y1, y2 = int(col.min().item()), int(col.max().item()) + 1
    return x1, y1, x2, y2


PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)


def _estimate_background(rgba: torch.Tensor, alpha_thr: float = 0.1) -> Tuple[float, float, float]:
    """Estimate background color from low-alpha pixels (median). Returns (r,g,b) in [0,1]."""
    rgb = rgba[:3].clamp(0, 1)
    alpha = rgba[3:4].clamp(0, 1).squeeze(0)
    low_alpha = alpha < alpha_thr
    n = low_alpha.sum().item()
    if n < 10:
        return PIXEL_MEAN
    # rgb: [3,H,W], low_alpha: [H,W]
    vals = rgb[:, low_alpha]  # [3, N]
    med = torch.median(vals, dim=1).values
    return tuple(med.detach().cpu().numpy().tolist())


def _get_fill_rgb(fill) -> Tuple[float, float, float]:
    """Resolve fill to (r,g,b) in 0-255 for crop/pad."""
    if fill == "mean":
        return tuple(255.0 * c for c in PIXEL_MEAN)
    if isinstance(fill, (tuple, list)) and len(fill) == 3:
        return tuple(255.0 * float(c) for c in fill)
    return (0.0, 0.0, 0.0)


def _crop_with_mask(image: torch.Tensor, mask: torch.Tensor, bbox, fill, expand_ratio: float = 1.0):
    if isinstance(bbox, torch.Tensor):
        bbox = tuple(bbox.flatten().tolist())
    l, t, r, b = _expand_box(*bbox, expand_ratio)
    _, h, w = image.shape
    l, t = max(l, 0), max(t, 0)
    r, b = min(r, w), min(b, h)
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    mask_crop = mask[t:b, l:r]
    fill_rgb = _get_fill_rgb(fill)
    fill_t = torch.stack([image.new_full((b - t, r - l), fill_value=v) for v in fill_rgb])
    region = image[:, t:b, l:r] * mask_crop.unsqueeze(0) + (1 - mask_crop.unsqueeze(0)) * fill_t
    return region


def _resize_with_padding(img: torch.Tensor, size: int = 224, fill_color=None) -> torch.Tensor:
    """Pad to square preserving aspect ratio, then resize to size x size."""
    c, h, w = img.shape
    side = max(h, w)
    fill = fill_color if fill_color is not None else PIXEL_MEAN
    if isinstance(fill, (tuple, list)) and len(fill) == 3:
        pad_vals = [float(x) for x in fill]
    else:
        pad_vals = list(PIXEL_MEAN)
    pad_t = (side - h) // 2
    pad_l = (side - w) // 2
    padded = img.new_zeros(c, side, side)
    for i in range(c):
        padded[i].fill_(pad_vals[i])
    padded[:, pad_t:pad_t + h, pad_l:pad_l + w] = img
    return F.interpolate(padded.unsqueeze(0), size=(size, size), mode="bicubic").squeeze(0)


def roi_crop_from_alpha(rgba: torch.Tensor, mask_thr: float = 0.05, expand_ratio: float = 1.2, fill: str = "mean", size: int = 224, min_pixels: int = 20, pad_resize: bool = True, bg_alpha_thr: float = 0.1):
    """ROI crop using alpha as mask (OVSeg-style). Returns [3, size, size] in [0,1].
    fill: 'mean' (CLIP gray), 'background' (estimate from low-alpha pixels), or (r,g,b) in [0,1].
    """
    rgb = rgba[:3].clamp(0, 1)
    alpha = rgba[3:4].clamp(0, 1)
    rgb_comp = (rgb * alpha).clamp(0, 1)
    image_255 = (rgb_comp * 255.0).to(rgba.dtype)
    mask = alpha.squeeze(0)
    bin_mask = (mask > mask_thr).float()

    fill_val = _estimate_background(rgba, bg_alpha_thr) if fill == "background" else fill
    fill_color = fill_val if isinstance(fill_val, (tuple, list)) and len(fill_val) == 3 else None

    if bin_mask.sum().item() < min_pixels:
        x = rgb_comp
        return _resize_with_padding(x, size, fill_color) if pad_resize else F.interpolate(x.unsqueeze(0), size=(size, size), mode="bicubic").squeeze(0)
    bbox = _mask_to_bbox(bin_mask)
    if bbox is None:
        x = rgb_comp
        return _resize_with_padding(x, size, fill_color) if pad_resize else F.interpolate(x.unsqueeze(0), size=(size, size), mode="bicubic").squeeze(0)
    region = _crop_with_mask(image_255, bin_mask, bbox, fill=fill_val, expand_ratio=expand_ratio)
    region_01 = (region / 255.0).clamp(0, 1)
    return _resize_with_padding(region_01, size, fill_color) if pad_resize else F.interpolate(region_01.unsqueeze(0), size=(size, size), mode="bicubic").squeeze(0)


# ----------------------------
# Orbit pose sampler
# ----------------------------
def _orbit_to_w2v(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    # Rasterizer culls p_view.z <= 0.2, so +Z must point into scene (camera looks down +Z)
    fwd = target - cam_pos
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(up, fwd)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(fwd, right)
    R_c2w = np.eye(4)
    R_c2w[:3, 0] = right
    R_c2w[:3, 1] = up
    R_c2w[:3, 2] = fwd  # +fwd: camera +Z into scene (rasterizer expects p_view.z > 0.2)
    R_c2w[:3, 3] = cam_pos
    w2c = np.linalg.inv(R_c2w)
    R = w2c[:3, :3].T
    t = w2c[:3, 3]
    return getWorld2View2(R, t)


@dataclass
class SyntheticPoseSampler:
    width: int = 800
    height: int = 800
    fov_deg: float = 50.0
    znear: float = 0.01
    zfar: float = 100.0
    elev_range_deg: Tuple[float, float] = (-30.0, 30.0)
    azim_range_deg: Tuple[float, float] = (-180.0, 180.0)
    alpha_min: float = 0.001  # looser for small objects in frame
    max_tries: int = 50
    no_reject: bool = False  # if True, accept all poses (no alpha check)
    target_center: Optional[np.ndarray] = None  # orbit around this point; if None, use bbox center of gs
    radius_fixed: Optional[Tuple[float, float]] = None  # if set, use instead of auto radius
    device: str = "cuda"

    def _make_camera(self, w2v: np.ndarray) -> Any:
        fov_rad = math.radians(self.fov_deg)
        fovx = fovy = fov_rad
        proj = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=fovx, fovY=fovy)
        w2v_t = torch.tensor(w2v, dtype=torch.float32, device=self.device).transpose(0, 1)
        proj_t = proj.transpose(0, 1).to(self.device)
        full_proj = w2v_t.unsqueeze(0).bmm(proj_t.unsqueeze(0)).squeeze(0)
        return MiniCam(
            width=self.width, height=self.height,
            fovx=fovx, fovy=fovy, znear=self.znear, zfar=self.zfar,
            world_view_transform=w2v_t, full_proj_transform=full_proj,
        )

    def _get_target_and_extent(self, gs: GaussianModel) -> Tuple[np.ndarray, float]:
        """Orbit center (bbox center) and extent (max bbox dimension)."""
        if self.target_center is not None:
            return np.array(self.target_center, dtype=np.float32), 0.5
        xyz = gs.get_xyz.detach().cpu().numpy()
        if xyz.shape[0] == 0:
            return np.zeros(3, dtype=np.float32), 0.5
        mn, mx = xyz.min(axis=0), xyz.max(axis=0)
        cen = ((mn + mx) / 2).astype(np.float32)
        ext = float(np.max(mx - mn))
        return cen, max(ext, 1e-6)

    def _get_target(self, gs: GaussianModel) -> np.ndarray:
        return self._get_target_and_extent(gs)[0]

    def _get_radius_range(self, gs: GaussianModel) -> Tuple[float, float]:
        """Radius scaled by extent so camera is outside object (unless overridden)."""
        if self.radius_fixed is not None:
            return self.radius_fixed
        _, ext = self._get_target_and_extent(gs)
        r_min = max(ext * 1.2, 0.5)
        r_max = max(ext * 2.5, r_min + 0.5)
        return (r_min, r_max)

    def _camera_from_orbit(
        self,
        r: float,
        el_deg: float,
        az_deg: float,
        target: np.ndarray,
    ) -> Any:
        up = np.array([0, 0, -1], dtype=np.float32)
        el = math.radians(el_deg)
        az = math.radians(az_deg)
        x = r * math.cos(el) * math.cos(az)
        y = r * math.cos(el) * math.sin(az)
        z = r * math.sin(el)
        cam_pos = target + np.array([x, y, z], dtype=np.float32)
        w2v = _orbit_to_w2v(cam_pos, target, up)
        view = self._make_camera(w2v)

        class CamWrap:
            pass

        c = CamWrap()
        c.view = view
        c.orbit_r = float(r)
        c.orbit_el_deg = float(el_deg)
        c.orbit_az_deg = float(az_deg)
        c.orbit_target = np.array(target, dtype=np.float32)
        return c

    def opposite_of(self, cam: Any) -> Optional[Any]:
        """Same radius/elevation, azimuth + 180 deg (other side of the object)."""
        r = getattr(cam, "orbit_r", None)
        el_deg = getattr(cam, "orbit_el_deg", None)
        az_deg = getattr(cam, "orbit_az_deg", None)
        target = getattr(cam, "orbit_target", None)
        if r is None or el_deg is None or az_deg is None or target is None:
            return None
        az_opp = float(az_deg) + 180.0
        if az_opp > 180.0:
            az_opp -= 360.0
        return self._camera_from_orbit(float(r), float(el_deg), az_opp, target)

    def sample_one(self, gs: GaussianModel, render_rgba_fn) -> Optional[Any]:
        target = self._get_target(gs)
        r_range = self._get_radius_range(gs)

        for _ in range(self.max_tries):
            r = np.random.uniform(*r_range)
            el_deg = np.random.uniform(*self.elev_range_deg)
            az_deg = np.random.uniform(*self.azim_range_deg)
            c = self._camera_from_orbit(float(r), float(el_deg), float(az_deg), target)
            if self.no_reject:
                return c
            rgba = render_rgba_fn(c, gs)
            alpha = rgba[3:4].clamp(0, 1)
            if alpha.mean().item() >= self.alpha_min:
                return c
        return None

    def sample(self, n: int, gs: GaussianModel, render_rgba_fn) -> List[Any]:
        out = []
        n_tries = n * 3 if not self.no_reject else n  # with no_reject, one pose per try
        for _ in range(n_tries):
            if len(out) >= n:
                break
            c = self.sample_one(gs, render_rgba_fn)
            if c is not None:
                c.id = len(out)
                out.append(c)
        return out

    def sample_novel_and_opposite(
        self, n_novel: int, gs: GaussianModel, render_rgba_fn
    ) -> List[Any]:
        """n_novel random views, each followed by its azimuth-opposite camera."""
        out: List[Any] = []
        n_tries = n_novel * 3 if not self.no_reject else n_novel
        for _ in range(n_tries):
            if len(out) >= 2 * n_novel:
                break
            c = self.sample_one(gs, render_rgba_fn)
            if c is None:
                continue
            opp = self.opposite_of(c)
            if opp is None:
                continue
            c.id = len(out)
            out.append(c)
            opp.id = len(out)
            out.append(opp)
        return out


# ----------------------------
# Renderer
# ----------------------------
class SimpleRenderer:
    def __init__(self, model_path: str, device: str = "cuda"):
        import argparse
        ap = argparse.ArgumentParser()
        ModelParams(ap, sentinel=True)
        pp = PipelineParams(ap)
        pipe_ns = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        self._pipe = pp.extract(pipe_ns)
        self._background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        self._bg_is_gray = False
        self.device = device
        self._scale_modifier = 1.0

    def set_background(self, color: str = "black"):
        """Set render background: 'black', 'gray' (CLIP mean), or 'white'."""
        if color == "gray":
            self._background = torch.tensor(list(PIXEL_MEAN), dtype=torch.float32, device=self._background.device)
            self._bg_is_gray = True
        elif color == "white":
            self._background = torch.tensor([1, 1, 1], dtype=torch.float32, device=self._background.device)
            self._bg_is_gray = False
        else:
            self._background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self._background.device)
            self._bg_is_gray = False

    def set_scale_modifier(self, v: float):
        self._scale_modifier = v

    def render_rgba(self, cam: Any, gaussians: GaussianModel, override_opacity=None) -> torch.Tensor:
        view = cam.view
        out = render(view, gaussians, self._pipe, self._background, scaling_modifier=self._scale_modifier, override_opacity=override_opacity)
        rgb = out["render"][:3].clamp(0, 1)
        if hasattr(self, "_bg_is_gray") and self._bg_is_gray:
            # Gray bg: alpha = 0 where pixel is close to background color
            bg = torch.tensor(PIXEL_MEAN, device=rgb.device, dtype=rgb.dtype).view(3, 1, 1)
            dist = (rgb - bg).abs().sum(dim=0, keepdim=True)
            alpha = (dist > 0.08).float()
        elif rgb.dim() == 3 and out["render"].shape[0] >= 4:
            alpha = out["render"][3:4].clamp(0, 1)
        else:
            alpha = (rgb.norm(dim=0, keepdim=True) > 0.02).float()
        return torch.cat([rgb, alpha.clamp(0, 1)], dim=0)


def main():
    parser = argparse.ArgumentParser(description="Render normalized PLY with synthetic orbit poses")
    parser.add_argument("-i", "--input_ply", required=True, help="Input PLY (normalized)")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for rendered images")
    parser.add_argument("-m", "--model_path", default=".", help="Path for renderer (e.g. output/exp/xyz)")
    parser.add_argument("--num_views", type=int, default=20, help="Number of views to render")
    parser.add_argument("--radius_min", type=float, default=None, help="Orbit radius min (default: auto from extent)")
    parser.add_argument("--radius_max", type=float, default=None, help="Orbit radius max (default: auto from extent)")
    parser.add_argument("--alpha_min", type=float, default=0.001, help="Min mean alpha for view acceptance")
    parser.add_argument("--no_reject", action="store_true", help="Accept all poses (skip alpha check)")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--save_rgba", action="store_true", help="Save RGBA (4ch) as .npy instead of RGB PNG")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scale_modifier", type=float, default=None,
                        help="Scale gaussians (default: auto for normalized extent 0.5 -> ~20)")
    parser.add_argument("--clip_crop", action="store_true",
                        help="Apply ROI crop (alpha mask) for CLIP-ready 224x224 output")
    parser.add_argument("--no_clip_crop", action="store_true",
                        help="Save full render only (no ROI crop)")
    parser.add_argument("--roi_mask_thr", type=float, default=0.05,
                        help="Alpha threshold for ROI mask (higher = exclude low-opacity pixels that rarely affect render)")
    parser.add_argument("--roi_expand_ratio", type=float, default=1.2,
                        help="ROI bbox padding (1.2 = 20%% larger)")
    parser.add_argument("--roi_fill", type=str, default="background", choices=["mean", "background"],
                        help="Fill for mask-out regions: 'mean' (CLIP gray) or 'background' (estimate from low-alpha pixels)")
    parser.add_argument("--bg_color", type=str, default="gray", choices=["black", "gray", "white"],
                        help="Render background: 'gray' (CLIP mean), 'white', or 'black'")
    args = parser.parse_args()

    if not os.path.isfile(args.input_ply):
        print(f"Error: input not found: {args.input_ply}", file=sys.stderr)
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    gs = GaussianModel(3)
    gs.load_ply(args.input_ply)
    print(f"[load] {args.input_ply}: {gs.get_xyz.shape[0]} points")

    scale_mod = args.scale_modifier if args.scale_modifier is not None else 1.0
    renderer = SimpleRenderer(model_path=args.model_path, device=device)
    renderer.set_background(args.bg_color)
    renderer.set_scale_modifier(scale_mod)
    radius_fixed = None
    if args.radius_min is not None and args.radius_max is not None:
        radius_fixed = (args.radius_min, args.radius_max)
    sampler = SyntheticPoseSampler(
        width=args.width, height=args.height,
        radius_fixed=radius_fixed,
        alpha_min=args.alpha_min,
        no_reject=args.no_reject,
        device=device,
    )

    def render_fn(cam, gaussians):
        return renderer.render_rgba(cam, gaussians)

    target, extent = sampler._get_target_and_extent(gs)
    r_min, r_max = sampler._get_radius_range(gs)
    print(f"[center] target={target}, extent={extent:.4f}, radius_range=({r_min:.2f}, {r_max:.2f})")

    cams = sampler.sample(args.num_views, gs, render_fn)
    print(f"[pose] sampled {len(cams)} valid views")

    if len(cams) == 0:
        print("Warning: no valid views (all rejected). Try loosening radius_range or alpha_min.", file=sys.stderr)
        sys.exit(1)

    use_clip_crop = args.clip_crop or not args.no_clip_crop
    print(f"[render] background={args.bg_color}")
    if use_clip_crop:
        print(f"[ROI crop] mask_thr={args.roi_mask_thr}, expand_ratio={args.roi_expand_ratio}, fill={args.roi_fill} -> 224x224 CLIP-ready")

    import torchvision
    for i, cam in enumerate(cams):
        rgba = renderer.render_rgba(cam, gs)
        if args.save_rgba:
            path = os.path.join(args.output_dir, f"view_{i:04d}.npy")
            np.save(path, rgba.detach().cpu().numpy())
        else:
            if use_clip_crop:
                region_224 = roi_crop_from_alpha(
                    rgba,
                    mask_thr=args.roi_mask_thr,
                    expand_ratio=args.roi_expand_ratio,
                    fill=args.roi_fill,
                    size=224,
                )
                path = os.path.join(args.output_dir, f"view_{i:04d}_clip.png")
                torchvision.utils.save_image(region_224.unsqueeze(0), path)
                path_full = os.path.join(args.output_dir, f"view_{i:04d}.png")
                torchvision.utils.save_image(rgba[:3].clamp(0, 1).unsqueeze(0), path_full)
                print(f"  saved {path} (CLIP) + {path_full} (full)")
            else:
                rgb = rgba[:3].clamp(0, 1)
                path = os.path.join(args.output_dir, f"view_{i:04d}.png")
                torchvision.utils.save_image(rgb.unsqueeze(0), path)
                print(f"  saved {path}")

    print(f"[done] {len(cams)} images -> {args.output_dir}")


if __name__ == "__main__":
    main()
