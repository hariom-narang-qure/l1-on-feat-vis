"""Cross-view label agreement: fraction of view-B line voxels that fall on an annotated slice of view A
and lie within 2 / 4 px (in A's plane) of A's line on that slice. Inside lung only."""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage
from l1_on_feat_vis.fissure_probe.data.views import ScanViews, scan_folders, short_id, annotated_slices, VIEW_AXIS, plane

def agree(folder):
    v = ScanViews(folder); lung = ndimage.binary_dilation(v.lobes > 0, iterations=3)
    out = {}
    for a in ("axial", "coronal", "sagittal"):
        ax = VIEW_AXIS[a]
        ks = [k for k in annotated_slices(folder, a)]
        dist = {}
        for k in ks:
            la = v.lines[a][plane(a, k)] & lung[plane(a, k)]
            dist[k] = ndimage.distance_transform_edt(~la) if la.any() else None
        for b in ("axial", "coronal", "sagittal"):
            if a == b: continue
            pts = np.argwhere(v.lines[b] & lung)
            pts = pts[np.isin(pts[:, ax], ks)]
            d = []
            for p in pts:
                D = dist[p[ax]]
                if D is None: d.append(np.inf); continue
                q = [p[i] for i in range(3) if i != ax]
                d.append(D[q[0], q[1]])
            d = np.array(d)
            out[(a, b)] = (len(d), (d <= 2).sum(), (d <= 4).sum())
    return short_id(folder), out

if __name__ == "__main__":
    with ProcessPoolExecutor(8) as pool:
        res = list(pool.map(agree, scan_folders()))
    tot = {}
    for s, o in res:
        for key, c in o.items():
            tot[key] = np.add(tot.get(key, 0), c)
    for (a, b), (n, w2, w4) in sorted(tot.items()):
        print(f"{b:9s} lines on {a:9s} slices: {n:6d} voxels, within 2 px of the {a} line {w2/n:.3f}, within 4 px {w4/n:.3f}")
