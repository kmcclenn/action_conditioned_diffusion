"""Sweep CFG guidance scales for each action-diffusion head and plot results.

Evaluates on the *first 2-second window* of each test clip
(`WindowDataset(deterministic=True)` returns `windows[0]` per clip), so every
head/guidance combo is scored on the same conditioning + ground-truth pair.

Metrics come from `action_diffusion.evaluate_sampling`:
  - action_mse_{mean,std}    over n_seeds  (denormalized 6D twist MSE)
  - trans_err_{mean,std}     translation L2 of integrated final pose (m)
  - rot_err_deg_{mean,std}   rotation angle of integrated final pose (deg)

Outputs into --out_dir:
  results.json
  action_mse.png  trans_err.png  rot_err.png    (skipped if matplotlib missing)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action_diffusion import (
    Diffusion,
    MLPActionDenoiser,
    TransformerActionDenoiser,
    WindowDataset,
    collate_windows,
    evaluate_sampling,
)


def build_model(head: str, sample_feat: torch.Tensor, ckpt_path: str | Path,
                device: torch.device) -> torch.nn.Module:
    if head == "mlp":
        model = MLPActionDenoiser(n_actions=8, feat_dim=sample_feat.shape[-1])
    elif head == "transformer":
        model = TransformerActionDenoiser(
            n_actions=8,
            feat_dim=sample_feat.shape[-1],
            n_tokens_per_img=sample_feat.shape[-2],
        )
    else:
        raise ValueError(head)
    # Action stats (action_mean / action_std buffers) live in the state dict.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"  loaded {ckpt_path} (epoch={ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f})")
    return model.to(device)


def plot_results(results: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("action_mse",  "action_mse_mean",  "action_mse_std",
         "Action MSE (denormalized)"),
        ("trans_err",   "trans_err_mean",   "trans_err_std",
         "Translation error (m)"),
        ("rot_err",     "rot_err_deg_mean", "rot_err_deg_std",
         "Rotation error (deg)"),
    ]
    ws = results["guidance_scales"]
    heads = list(results["heads"].keys())
    x = np.arange(len(ws))
    width = 0.8 / max(len(heads), 1)

    for fname, mkey, skey, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, head in enumerate(heads):
            per_w = results["heads"][head]["per_guidance"]
            means = [r[mkey] for r in per_w]
            stds  = [r[skey] for r in per_w]
            offset = (i - (len(heads) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=head)
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in ws])
        ax.set_xlabel("CFG guidance scale w")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs CFG (n_seeds={results['n_seeds']}, "
                     f"n_clips={results['n_test_clips']})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname}.png", dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",           default="RealEstate10K")
    p.add_argument("--features_root",  default="features_croco")
    p.add_argument("--mlp_ckpt",       required=True)
    p.add_argument("--tf_ckpt",        required=True)
    p.add_argument("--guidance_scales", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, 3.0])
    p.add_argument("--n_seeds",        type=int, default=16)
    p.add_argument("--batch_size",     type=int, default=256)
    p.add_argument("--num_workers",    type=int, default=0)
    p.add_argument("--num_steps",      type=int, default=100,
                   help="DDPM K (must match training).")
    p.add_argument("--out_dir",        default="eval/action_diffusion")
    p.add_argument("--device",         default=None)
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # First 2 s of each test clip: deterministic=True returns windows[0].
    val_ds = WindowDataset(
        args.root, args.features_root, split="test",
        n_actions=8, action_hz=4.0, window_stride_seconds=0.5,
        deterministic=True,
    )
    print(f"test clips with valid first 2s window: {len(val_ds)}")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_windows,
    )

    sample_feat = val_ds._clip_features[0][0]   # (N_tok, feat_dim)
    diffusion = Diffusion(num_steps=args.num_steps).to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "guidance_scales": list(args.guidance_scales),
        "n_seeds":         args.n_seeds,
        "n_test_clips":    len(val_ds),
        "num_steps":       args.num_steps,
        "heads":           {},
    }

    for head, ckpt_path in [("mlp", args.mlp_ckpt),
                            ("transformer", args.tf_ckpt)]:
        print(f"\n=== {head}  ({ckpt_path}) ===")
        model = build_model(head, sample_feat, ckpt_path, device)
        per_w = []
        for w in args.guidance_scales:
            print(f"  w={w}  sampling...", flush=True)
            r = evaluate_sampling(
                model, diffusion, val_loader, device,
                guidance_scale=w, n_seeds=args.n_seeds,
            )
            print(f"    action_mse {r['action_mse_mean']:.4f} ± {r['action_mse_std']:.4f}   "
                  f"trans_err {r['trans_err_mean']:.4f} ± {r['trans_err_std']:.4f} m   "
                  f"rot_err {r['rot_err_deg_mean']:.2f} ± {r['rot_err_deg_std']:.2f} deg")
            per_w.append(r)
        results["heads"][head] = {
            "ckpt":          str(ckpt_path),
            "per_guidance":  per_w,
        }

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {json_path}")

    try:
        plot_results(results, out_dir)
        print(f"wrote bar charts to {out_dir}/")
    except ImportError:
        print("[warn] matplotlib not installed — JSON saved, plots skipped. "
              "Install matplotlib and re-run plot_results(json.load(...), out_dir).")


if __name__ == "__main__":
    main()
