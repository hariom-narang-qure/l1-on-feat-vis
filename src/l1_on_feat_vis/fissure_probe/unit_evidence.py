"""Phase 7: evidence for the selected units.

1. Pretraining control. Each unit is re-read from a random-init twin of its
   network (same architecture, seed 0; nnU-Net: same plans, reset weights) at
   the same store rows of EVIDENCE_SCANS (per fold one review_v1 + one severe train scan),
   and scored like cv_scores.py on those rows (per scan-fold, min over folds):
   line-vs-ring AUROC and natural-frequency AP lift. If the twin scores as well,
   the "fissure unit" is architecture + location, not learned.
2. Top activations. The 16 store rows (all 12 train scans) where the signed
   unit is highest, as 48 px canvas crops around the token with the agent's
   line in red: what the unit actually fires on.
"""

import argparse
import json

import numpy as np
import timm
import torch
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoModel

from l1_on_feat_vis.fissure_probe.cv_scores import RowMasks, load_scores, weighted_ap
from l1_on_feat_vis.fissure_probe.nnunet_features import NETWORKS, NnUNet
from l1_on_feat_vis.fissure_probe.rows import FOLDS, Rows
from l1_on_feat_vis.fissure_probe.store import (MODELS_2D, Conv, Taps3D, Vit, canvas_grey, load_layer, model_input,
                                                scan_folder, token_centres)
from l1_on_feat_vis.fissure_probe.unit_scan import auroc_columns
from l1_on_feat_vis.fissure_probe.views import BITS, STORE, VIEWS, ScanViews

def evidence_scans():
    """One review_v1 and one severe train scan per fold, so every fold has rows."""
    from l1_on_feat_vis.fissure_probe.data import list_scans
    from l1_on_feat_vis.fissure_probe.probes import split_scans
    from l1_on_feat_vis.fissure_probe.rows import fold_of_scans
    train = split_scans()[0]
    folds, cohort = fold_of_scans(train), {r.short: r.cohort for r in list_scans()}
    return tuple(next(s for s in train if folds[s] == f and cohort[s] == c) for f in range(FOLDS) for c in ("review_v1", "severe"))


EVIDENCE_SCANS = evidence_scans()
CROP = 48
EVIDENCE = STORE / "evidence"


class RandomVit(Vit):
    def load(self, device):
        torch.manual_seed(0)
        self.model = AutoModel.from_config(AutoConfig.from_pretrained(self.repo)).to(device).eval()
        self.skip = 1 + getattr(self.model.config, "num_register_tokens", 0)
        self.mlp_out = {}
        if self.with_mlp:
            for l, layer in enumerate(self.model.encoder.layer):
                layer.mlp.activation_fn.register_forward_hook(lambda m, i, o, l=l: self.mlp_out.__setitem__(l, o))


class RandomConv(Conv):
    def load(self, device):
        torch.manual_seed(0)
        self.model = timm.create_model(self.timm_name, pretrained=False).to(device).eval()
        self.taps = {}
        for prefix, blocks in self.block_lists(self.model):
            for b, block in enumerate(blocks):
                block.register_forward_hook(lambda m, i, o, key=f"{prefix}b{b}": self.taps.__setitem__(key, o))


class RandomTaps3D(Taps3D):
    def __init__(self, name, device):
        super().__init__(name, device)
        torch.manual_seed(0)
        for module in self.net.network.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


def random_twin(model, device):
    spec = MODELS_2D[model]
    twin = (RandomVit(spec.name, spec.repo, spec.mean, spec.std, spec.with_mlp) if isinstance(spec, Vit)
            else RandomConv(spec.name, spec.timm_name, spec.block_lists))
    twin.load(device)
    return twin


def twin_values(units, rows, device):
    """{(layer, unit): values at the rows of EVIDENCE_SCANS (row-table order)} from random-init twins."""
    by_model = {}
    for layer, unit in units:
        by_model.setdefault(layer.split("/")[0], []).append((layer, unit))
    out = {key: [] for key in units}
    for model, members in by_model.items():
        if model in MODELS_2D:
            twin = random_twin(model, device)
        else:
            twin = RandomTaps3D(model, device)
        for scan in EVIDENCE_SCANS:
            index = rows.of_scan(rows.scans.index(scan))
            slices, tokens = rows.slice[index], rows.token[index]
            cache = np.load(STORE / "slices" / f"{scan}.npz")
            values = {key: np.zeros(len(index), np.float32) for key in members}
            if model in MODELS_2D:
                for s in np.unique(slices):
                    these = slices == s
                    x = model_input(canvas_grey(cache["hu"][s]), twin.mean, twin.std, twin.input_px, device)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        layers = twin.layers(x, torch.from_numpy(tokens[these].astype(np.int64)).to(device))
                    for layer, unit in members:
                        values[(layer, unit)][these] = layers[layer.split("/")[1]][:, unit].float().cpu().numpy()
            else:
                views = ScanViews(scan_folder(scan))
                p, q = token_centres(tokens)
                voxels = np.zeros((len(tokens), 3))
                for s in np.unique(slices):
                    these = slices == s
                    voxels[these] = views.geometry[VIEWS[cache["view"][s]]].canvas_to_voxel(p[these], q[these], int(cache["k"][s]))
                wanted = {}
                for layer, unit in members:
                    wanted.setdefault(layer.split("/")[1], []).append(unit)
                layers = twin.layers(views.hu, views.spacing, views.lobes, voxels, wanted)
                for name, units_here in wanted.items():
                    for column, unit in enumerate(units_here):
                        values[(f"{model}/{name}", unit)] = layers[name][:, column].float().numpy()
            for key in members:
                out[key].append(values[key])
        del twin
        torch.cuda.empty_cache()
    return {key: np.concatenate(parts) for key, parts in out.items()}


def fold_scores(x, rows, index, device):
    """(min-fold line-vs-ring AUROC, min-fold AP lift) of one signed column on rows[index], folds = scan folds."""
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    line, ring, fold = rows.line[index], rows.ring[index], rows.fold[index]
    natural = (rows.line | rows.in_uniform)[index]
    weight = np.where(rows.line, 1.0, rows.weight)[index]
    aucs, lifts = [], []
    for f in range(FOLDS):
        m = (fold == f) & (line | ring)
        aucs.append(float(auroc_columns(g(x[m])[:, None], g(line[m]))[0]))
        n = (fold == f) & natural
        prior = (weight[n] * line[n]).sum() / weight[n].sum()
        lifts.append(float(weighted_ap(g(x[n])[:, None], g(line[n]), g(weight[n]))[0]) / prior)
    return min(aucs), min(lifts)


def top_sheet(entries, rows, out_path, count=16):
    tiles = []
    caches = {}
    for e in entries:
        x = load_layer(e["layer"], rows.scans, [e["unit"]])[:, 0] * e["sign"]
        row_tiles = []
        for r in np.argsort(-x)[:count]:
            scan = rows.scans[rows.scan[r]]
            cache = caches.setdefault(scan, np.load(STORE / "slices" / f"{scan}.npz"))
            p, q = (int(round(float(v[0]))) for v in token_centres(np.array([rows.token[r]])))
            s = rows.slice[r]
            hu = np.pad(cache["hu"][s], CROP, constant_values=-1024)[p:p + 2 * CROP, q:q + 2 * CROP]
            line = np.pad((cache["labels"][s] & BITS["line"]) > 0, CROP)[p:p + 2 * CROP, q:q + 2 * CROP]
            rgb = np.repeat((canvas_grey(hu) * 255).astype(np.uint8)[..., None], 3, -1)
            rgb[line] = [255, 40, 40]
            row_tiles.append(Image.fromarray(rgb).resize((96, 96), Image.NEAREST))
        tiles.append((f"{e['layer']} {e['unit']}", row_tiles))
    out = Image.new("RGB", (220 + count * 98, len(tiles) * 98), "black")
    draw = ImageDraw.Draw(out)
    for i, (label, row_tiles) in enumerate(tiles):
        draw.text((4, i * 98 + 40), label, fill=(255, 255, 0))
        for j, tile in enumerate(row_tiles):
            out.paste(tile, (220 + j * 98, i * 98))
    out.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--count", type=int, default=16, help="first units of the pool (entry order)")
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    rows = Rows()
    entries = json.loads(open(args.pool).read())["pool"][:args.count]
    index = np.concatenate([rows.of_scan(rows.scans.index(s)) for s in EVIDENCE_SCANS])
    twins = twin_values([(e["layer"], e["unit"]) for e in entries], rows, device)
    report = []
    for e in entries:
        pretrained = load_layer(e["layer"], list(EVIDENCE_SCANS), [e["unit"]])[:, 0] * e["sign"]
        twin = twins[(e["layer"], e["unit"])]
        twin_sign = 1 if fold_scores(twin, rows, index, device)[0] >= fold_scores(-twin, rows, index, device)[0] else -1
        p_auc, p_lift = fold_scores(pretrained, rows, index, device)
        r_auc, r_lift = fold_scores(twin_sign * twin, rows, index, device)
        report.append({**e, "pretrained_ring_auc": p_auc, "pretrained_ap_lift": p_lift,
                       "random_ring_auc": r_auc, "random_ap_lift": r_lift})
        print(f"{e['layer']:20s} {e['unit']:5d} | pretrained min-fold ring AUROC {p_auc:.3f} AP lift {p_lift:4.1f}x | "
              f"random-init {r_auc:.3f} {r_lift:4.1f}x", flush=True)
    (EVIDENCE / "pretraining_control.json").write_text(json.dumps(report, indent=1, default=float))
    top_sheet(entries, rows, "outputs/fissure_rerun/top_activations.png")


if __name__ == "__main__":
    main()
