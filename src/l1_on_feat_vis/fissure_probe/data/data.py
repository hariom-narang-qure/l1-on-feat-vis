"""Scans, fissure labels and patch sampling for probing.

Dataset: the 19 scans under fissure-annotation/work/scans/<uid>/flex_step4/
(9 "Fissure Segmentation Review v1" scans from 19 Oct, 10 severe scans from
16 Oct). Per scan:
  - scan.nii.gz: CT in HU (thick-slice scans were already resampled along z).
  - post/sheet.nii.gz: the waypoint agent's fissure sheet after post-processing.
    These are machine labels (LLM waypoints + a Hessian ridge tracer), so a
    Hessian sheetness filter is kept as a baseline.
  - lobes.nii.gz: TotalSegmentator lobes (labels 10-14). Their inter-lobe
    boundaries are a second fissure proxy, independent of the agent.

Label maps (all at full resolution, 1 voxel = one CT voxel):
  - fissure: the sheet, dilated by one voxel (3x3x3). The agent's lines are one
    voxel wide and off by a voxel often enough that exact-voxel labels would
    punish a detector for that.
  - lobe_boundary: lobe voxels with a 6-neighbour in a different lobe, dilated
    by one voxel.
  - pleura: lung voxels with a 6-neighbour outside the lung, dilated by one voxel.
  - ring: lung voxels 3-8 voxels from the sheet. Location-matched negatives:
    they share the fissure's surroundings (lung interior, near the hilum), so
    "fissure vs ring" asks whether a detector peaks on the fissure itself
    rather than on where fissures tend to be.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from pydantic import BaseModel
from scipy import ndimage

SCANS_ROOT = Path("/home/hariom.narang/fissure-annotation/work/scans")
HU_MIN, HU_MAX = -1000.0, 1000.0
CROSS = ndimage.generate_binary_structure(3, 1)
CUBE = ndimage.generate_binary_structure(3, 3)


class ScanRecord(BaseModel):
    uid: str
    cohort: str  # "review_v1" (19 Oct) or "severe" (16 Oct)
    folder: Path

    @property
    def short(self):
        return self.uid[-6:]


def cohort_of(folder):
    """The two cohorts are told apart by run date (status.json mtime)."""
    day = np.datetime64(int((folder / "status.json").stat().st_mtime), "s").astype("datetime64[D]")
    return "review_v1" if str(day) == "2026-10-19" else "severe"


def list_scans(root=SCANS_ROOT):
    folders = sorted(path for path in root.glob("*/flex_step4") if (path / "post" / "sheet.nii.gz").exists())
    return [ScanRecord(uid=folder.parent.name, cohort=cohort_of(folder), folder=folder) for folder in folders]


def hu_to_genesis_input(hu):
    return (np.clip(hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)


def dilate(mask):
    return ndimage.binary_dilation(mask, CUBE)


def outer_surface(mask):
    return mask & ~ndimage.binary_erosion(mask, CROSS)


class LoadedScan:
    def __init__(self, record):
        self.record = record
        folder = record.folder
        self.hu = np.asarray(nib.load(folder / "scan.nii.gz").dataobj, dtype=np.float32)
        self.lobes = np.asarray(nib.load(folder / "lobes.nii.gz").dataobj, dtype=np.uint8)
        sheet = np.asarray(nib.load(folder / "post" / "sheet.nii.gz").dataobj) > 0
        self.lung = self.lobes > 0
        self.sheet = sheet
        self.fissure = dilate(sheet)
        self.sheet_voxels = int(sheet.sum())
        self.lobe_boundary = dilate(self.inter_lobe_boundary())
        self.pleura = dilate(outer_surface(self.lung)) & ~self.lobe_boundary
        distance_to_sheet = ndimage.distance_transform_edt(~sheet)
        self.ring = self.lung & (distance_to_sheet >= 3) & (distance_to_sheet <= 8)

    def inter_lobe_boundary(self):
        """Lobe voxels with a 6-neighbour in a different lobe.

        Max-filter > own label: a higher lobe is next door. Min-filter (with
        background lifted to 255 so it never counts) < own label: a lower one is.
        """
        lobes = self.lobes
        neighbour_max = ndimage.grey_dilation(lobes, footprint=CROSS)
        background_high = np.where(lobes > 0, lobes, 255).astype(np.uint8)
        neighbour_min = ndimage.grey_erosion(background_high, footprint=CROSS)
        touches_higher = (neighbour_max > lobes) & (lobes > 0)
        touches_lower = (neighbour_min < lobes) & (lobes > 0)
        return touches_higher | touches_lower


class PatchSampler:
    """Patch corners inside the lung, half centred on fissure voxels.

    Patches are `shape` voxels. Half the centres are random fissure voxels so
    every patch set contains fissure; half are random lung voxels so the
    negatives cover vessels, airways and pleura in their natural proportion.
    """

    def __init__(self, shape=(128, 128, 64), seed=0, min_lung_fraction=0.5):
        self.shape = np.array(shape)
        self.rng = np.random.default_rng(seed)
        self.min_lung_fraction = min_lung_fraction

    def corners(self, scan, count):
        fissure_points = np.argwhere(scan.fissure & scan.lung)
        lung_points = np.argwhere(scan.lung)
        corners = []
        for index in range(count):
            points = fissure_points if index % 2 == 0 and len(fissure_points) else lung_points
            corners.append(self.mostly_lung_corner(scan, points))
        return corners

    def mostly_lung_corner(self, scan, points, attempts=50):
        """A patch around a random point that is at least `min_lung_fraction` lung
        (the best of `attempts` tries if none is), so patches are not mostly chest wall."""
        best, best_fraction = None, -1.0
        for _ in range(attempts):
            centre = points[self.rng.integers(len(points))]
            corner = np.clip(centre - self.shape // 2, 0, np.array(scan.hu.shape) - self.shape)
            fraction = self.crop(scan.lung, corner).mean()
            if fraction >= self.min_lung_fraction:
                return corner
            if fraction > best_fraction:
                best, best_fraction = corner, fraction
        return best

    def crop(self, volume, corner):
        x, y, z = corner
        dx, dy, dz = self.shape
        return volume[x:x + dx, y:y + dy, z:z + dz]
