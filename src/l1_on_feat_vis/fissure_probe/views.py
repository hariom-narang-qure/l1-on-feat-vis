"""Annotated slices of all three views as square-pixel 512 x 512 canvases with labels.

Ground truth: fissures_<view>.nii.gz, the waypoint agent's 1 px ridge-traced
lines, one file per view, on the grid of flex_step4/scan.nii.gz. A slice is
used only if the agent was shown it (its log exists) and it has a line;
line-free slices are left out for now (user decision, 2026-10-21).

Slice cut, exactly as the agent saw it (fissure_annotation/imaging/slices.py:55):
    image = volume[plane].T[::-1, ::-1]
  - axial:    rows = y reversed (anterior up), cols = x reversed (patient right left)
  - coronal:  rows = z reversed (superior up), cols = x reversed
  - sagittal: rows = z reversed (superior up), cols = y reversed (anterior left)
For coronal / sagittal the rows step by the z spacing and the cols by the
in-plane spacing; 11 of the 19 scans keep a native z that differs (z/x from
0.64 to 1.35), so their images are stretched.

Canvas (what 2D models see, and where labels and metrics live):
  1. Resample rows so they step by the col spacing (factor f = z / col spacing),
     pixel-centre aligned: original row r = (r' + 0.5) / f - 0.5.
     CT linear, lobes nearest. Axial slices are already square (f = 1).
  2. Cut a 512-row window centred on the scan's lung extent along the rows
     (pad with air when the image is shorter). A fixed 512 px canvas keeps one
     token = 512 / 74 = 6.9 px in every view; a max(H, W) canvas would make a
     875-slice scan's tokens twice as coarse.
  3. Lines are not interpolated (shrinking drops pixels, stretching leaves gaps):
     each pair of 8-adjacent line pixels is mapped and redrawn as a segment.

Labels on the canvas (bits of one uint8 map):
  - line:     the redrawn 1 px line, inside the lung only
  - positive: line dilated by 1 px (3 x 3), what the head is trained on
  - lung:     lobes > 0 grown by 3 px so the pleura is inside the scored area
Slices whose whole line lies outside the lung are dropped.
  - ring:     lung 3-8 px from the line (location-matched negatives)
  - pleura:   within 2 px of the lobes' outer edge, not positive
  - vessel:   inside the lobes, HU > -500, more than 2 px from the line, not pleura
  - lobes:    lobes > 0 itself
Every lung pixel that is not positive is a negative.
"""

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.draw import line as draw_line

SCANS_ROOT = Path("/home/hariom.narang/fissure-annotation/work/scans")
STORE = Path("/realdata/hariom.narang/fissure_acts")
VIEWS = ("axial", "coronal", "sagittal")
VIEW_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
CANVAS = 512
AIR_HU = -1024
LUNG_GROW_PX, PLEURA_BAND_PX = 3, 2
RING_PX = (3, 8)
VESSEL_HU = -500
BITS = {"line": 1, "positive": 2, "lung": 4, "ring": 8, "pleura": 16, "vessel": 32, "lobes": 64}


def plane(view, k):
    index = [slice(None)] * 3
    index[VIEW_AXIS[view]] = k
    return tuple(index)


def agent_view(volume, view, k):
    return volume[plane(view, k)].T[::-1, ::-1]


def annotated_slices(folder, view):
    logs = (folder / "fissures_work" / view / "run" / "logs").glob("slice_*.json")
    return sorted(int(re.search(r"slice_(\d+)", p.name).group(1)) for p in logs)


def row_col_spacing(spacing_xyz, view):
    sx, sy, sz = spacing_xyz
    return {"axial": (sy, sx), "coronal": (sz, sx), "sagittal": (sz, sy)}[view]


def view_shape(shape_xyz, view):
    """(rows, cols) of an agent-view image."""
    X, Y, Z = shape_xyz
    return {"axial": (Y, X), "coronal": (Z, X), "sagittal": (Z, Y)}[view]


class ViewGeometry:
    """Canvas pixel <-> agent-view pixel <-> voxel, for one view of one scan."""

    def __init__(self, shape_xyz, spacing_xyz, view, lung_rows):
        self.shape_xyz, self.view = shape_xyz, view
        row_spacing, col_spacing = row_col_spacing(spacing_xyz, view)
        self.factor = row_spacing / col_spacing
        self.pixel_mm = col_spacing
        rows, cols = view_shape(shape_xyz, view)
        self.rows, self.cols = rows, cols
        resampled_rows = int(round(rows * self.factor))
        lung_centre = (self.to_resampled_row(lung_rows[0]) + self.to_resampled_row(lung_rows[1])) / 2
        self.row_offset = self.window_start(resampled_rows, lung_centre)
        self.col_offset = self.window_start(cols, cols / 2)

    @staticmethod
    def window_start(length, centre):
        """First resampled index on the canvas: centred on `centre`, padded when shorter than the canvas."""
        if length <= CANVAS:
            return (length - CANVAS) // 2
        return int(np.clip(round(centre - CANVAS / 2), 0, length - CANVAS))

    def to_resampled_row(self, r):
        return (r + 0.5) * self.factor - 0.5

    def canvas_to_view(self, p, q):
        """Canvas (row, col) -> fractional agent-view (row, col)."""
        return (p + self.row_offset + 0.5) / self.factor - 0.5, q + self.col_offset

    def view_to_canvas(self, r, c):
        return self.to_resampled_row(r) - self.row_offset, c - self.col_offset

    def canvas_to_voxel(self, p, q, k):
        """Canvas (row, col) of slice k -> fractional voxel (x, y, z): the inverse of agent_view."""
        r, c = self.canvas_to_view(np.asarray(p, float), np.asarray(q, float))
        X, Y, Z = self.shape_xyz
        k = np.full_like(r, k)
        if self.view == "axial":
            return np.stack([X - 1 - c, Y - 1 - r, k], -1)
        if self.view == "coronal":
            return np.stack([X - 1 - c, k, Z - 1 - r], -1)
        return np.stack([k, Y - 1 - c, Z - 1 - r], -1)

    def resample(self, image, order, fill):
        p, q = np.meshgrid(np.arange(CANVAS), np.arange(CANVAS), indexing="ij")
        r, c = self.canvas_to_view(p, q)
        inside = (r >= -0.5) & (r <= self.rows - 0.5) & (c >= 0) & (c <= self.cols - 1)
        out = ndimage.map_coordinates(image.astype(np.float32), [np.clip(r, 0, self.rows - 1), c], order=order, mode="nearest")
        return np.where(inside, out, fill)

    def redraw_line(self, line):
        """Map every pair of 8-adjacent line pixels to the canvas and join them, so the line stays connected."""
        out = np.zeros((CANVAS, CANVAS), bool)
        rows, cols = np.nonzero(line)
        mapped = {(r, c): tuple(np.round(self.view_to_canvas(r, c)).astype(int)) for r, c in zip(rows, cols)}
        for (r, c), start in mapped.items():
            stroke(out, start, start)
            for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
                end = mapped.get((r + dr, c + dc))
                if end is not None:
                    stroke(out, start, end)
        return out


def stroke(canvas, start, end):
    rr, cc = draw_line(*start, *end)
    keep = (rr >= 0) & (rr < CANVAS) & (cc >= 0) & (cc < CANVAS)
    canvas[rr[keep], cc[keep]] = True


def disk_dilate(mask, radius):
    return ndimage.distance_transform_edt(~mask) <= radius


def slice_labels(hu, lobes, line):
    """The label bitmap of one canvas (see the module docstring).

    Line pixels outside the grown lung are dropped: 5.6% of all line pixels
    (120 whole slices) lie there, and on inspection they are agent errors on
    edge slices (chest wall, fat planes, the diaphragm dome).
    """
    lobe_mask = lobes > 0
    lung = disk_dilate(lobe_mask, LUNG_GROW_PX)
    line = line & lung
    positive = ndimage.binary_dilation(line, np.ones((3, 3), bool)) & lung
    distance_to_line = ndimage.distance_transform_edt(~line) if line.any() else np.full(line.shape, np.inf)
    ring = lung & (distance_to_line >= RING_PX[0]) & (distance_to_line <= RING_PX[1])
    outside, inside = ndimage.distance_transform_edt(~lobe_mask), ndimage.distance_transform_edt(lobe_mask)
    pleura = ((outside <= PLEURA_BAND_PX) & (outside > 0) | (lobe_mask & (inside <= PLEURA_BAND_PX))) & ~positive
    vessel = lobe_mask & (hu > VESSEL_HU) & (distance_to_line > 2) & ~pleura
    labels = np.zeros(line.shape, np.uint8)
    for name, mask in (("line", line), ("positive", positive), ("lung", lung), ("ring", ring),
                       ("pleura", pleura), ("vessel", vessel), ("lobes", lobe_mask)):
        labels |= np.where(mask, BITS[name], 0).astype(np.uint8)
    return labels


def has(labels, name):
    return (labels & BITS[name]) > 0


class ScanViews:
    """One scan's volumes and per-view geometry."""

    def __init__(self, folder):
        self.folder = folder
        image = nib.load(folder / "scan.nii.gz")
        self.spacing = tuple(float(s) for s in image.header.get_zooms()[:3])
        self.hu = np.asarray(image.dataobj, dtype=np.int16)
        self.lobes = np.asarray(nib.load(folder / "lobes.nii.gz").dataobj, dtype=np.uint8)
        self.lines = {view: np.asarray(nib.load(folder / f"fissures_{view}.nii.gz").dataobj) > 0 for view in VIEWS}
        self.geometry = {view: ViewGeometry(self.hu.shape, self.spacing, view, self.lung_rows(view)) for view in VIEWS}

    def lung_rows(self, view):
        """First and last agent-view row that contains lung, over the whole scan."""
        lung = self.lobes > 0
        rows_axis = {"axial": 1, "coronal": 2, "sagittal": 2}[view]
        present = np.flatnonzero(lung.any(axis=tuple(a for a in range(3) if a != rows_axis)))
        length = self.hu.shape[rows_axis]
        return length - 1 - present.max(), length - 1 - present.min()  # rows run reversed

    def line_slices(self, view):
        annotated = annotated_slices(self.folder, view)
        return [k for k in annotated if self.lines[view][plane(view, k)].any()]

    def canvas(self, view, k):
        geometry = self.geometry[view]
        hu = geometry.resample(agent_view(self.hu, view, k), order=1, fill=AIR_HU).astype(np.int16)
        lobes = geometry.resample(agent_view(self.lobes, view, k), order=0, fill=0).astype(np.uint8)
        line = geometry.redraw_line(agent_view(self.lines[view], view, k))
        return hu, lobes, line

    def agent_mask_matches(self, view, k):
        """Does our cut of fissures_<view> equal the agent's own mask PNG? (None if the agent saved none.)"""
        png = self.folder / "fissures_work" / view / "run" / "masks" / f"slice_{k:04d}_mask.png"
        if not png.exists():
            return None
        return bool(np.array_equal(np.asarray(Image.open(png)) > 0, agent_view(self.lines[view], view, k)))


def scan_folders(root=SCANS_ROOT):
    folders = [p for p in root.glob("*/flex_step4") if (p / "status.json").exists()]
    return sorted(folders, key=lambda p: p.parent.name[-6:])


def short_id(folder):
    return folder.parent.name[-6:]


def lost_line_pixels(views, view, k):
    """Line pixels outside the canvas window (would be silently dropped)."""
    geometry = views.geometry[view]
    rows, cols = np.nonzero(agent_view(views.lines[view], view, k))
    p, q = geometry.view_to_canvas(rows, cols)
    return int(((p < -0.5) | (p > CANVAS - 0.5) | (q < -0.5) | (q > CANVAS - 0.5)).sum())


def build_scan(folder, out_dir):
    """Cache every line slice of every view: canvas HU, labels and geometry. Returns check counts."""
    views = ScanViews(folder)
    hu, labels, view_ids, ks = [], [], [], []
    report = {"spacing": views.spacing, "mask_png_mismatch": 0, "mask_png_checked": 0, "lost_line_px": 0, "views": {}}
    for v, view in enumerate(VIEWS):
        slices = views.line_slices(view)
        geometry = views.geometry[view]
        report["views"][view] = {"annotated": len(annotated_slices(folder, view)), "line_slices": len(slices),
                                 "factor": geometry.factor, "row_offset": geometry.row_offset, "col_offset": geometry.col_offset}
        for k in slices:
            match = views.agent_mask_matches(view, k)
            if match is not None:
                report["mask_png_checked"] += 1
                report["mask_png_mismatch"] += int(not match)
            report["lost_line_px"] += lost_line_pixels(views, view, k)
            canvas_hu, canvas_lobes, canvas_line = views.canvas(view, k)
            slice_label = slice_labels(canvas_hu, canvas_lobes, canvas_line)
            if not has(slice_label, "line").any():
                report["views"][view]["dropped_line_outside_lung"] = report["views"][view].get("dropped_line_outside_lung", 0) + 1
                continue
            hu.append(canvas_hu)
            labels.append(slice_label)
            view_ids.append(v)
            ks.append(k)
    geometry = {f"{view}_{name}": getattr(views.geometry[view], name)
                for view in VIEWS for name in ("factor", "row_offset", "col_offset", "pixel_mm")}
    np.savez(out_dir / f"{short_id(folder)}.npz", hu=np.stack(hu), labels=np.stack(labels),
             view=np.array(view_ids, np.int8), k=np.array(ks, np.int16), shape_xyz=np.array(views.hu.shape), **geometry)
    return report


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=STORE / "slices")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folders = scan_folders()
    with ProcessPoolExecutor(args.workers) as pool:
        reports = dict(zip([short_id(f) for f in folders], pool.map(build_scan, folders, [args.out] * len(folders))))
    (args.out / "report.json").write_text(json.dumps(reports, indent=1))
    for scan, r in reports.items():
        counts = " ".join(f"{v[:3]} {r['views'][v]['line_slices']}/{r['views'][v]['annotated']} f={r['views'][v]['factor']:.2f}"
                          for v in VIEWS)
        print(f"{scan} {counts} | png mismatch {r['mask_png_mismatch']}/{r['mask_png_checked']} lost px {r['lost_line_px']}", flush=True)


if __name__ == "__main__":
    main()
