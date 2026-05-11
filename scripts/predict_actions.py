"""
Run a CroCo IDM on a video or directory of GIFs and predict SE(3) actions.

For each consecutive pair of frames the script:
  1. Extracts frames from the video/GIF.
  2. Preprocesses them with the same PREPROCESS transform used during training.
  3. Encodes them with the frozen CroCo V2 ViT-Base encoder.
  4. Passes the patch-token pairs through the chosen IDM (mp or tf).
  5. Denormalizes and returns the resulting 6D twists (v, omega).

Usage:
    python predict_actions.py --model mp --video tmp6gr0oup6.mp4
    python predict_actions.py --model tf --interp_dir /path/to/re10k-mini-interp_py
    python predict_actions.py --model mp --gif_dir /path/to/gen --output actions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, every_n: int = 1) -> list[Image.Image]:
    """Return a list of PIL RGB images sampled every `every_n` frames."""
    try:
        import cv2
    except ImportError:
        sys.exit("[error] OpenCV not found.  pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open video: {video_path}")

    frames: list[Image.Image] = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()
    return frames


def extract_gif_frames(gif_path: Path) -> list[Image.Image]:
    """Return all frames of a GIF as PIL RGB images."""
    gif = Image.open(gif_path)
    frames: list[Image.Image] = []
    for i in range(gif.n_frames):
        gif.seek(i)
        frames.append(gif.convert("RGB").copy())
    return frames


def predict_actions_for_frames(
    frames: list[Image.Image],
    forward_fn,
    idm,
    device: torch.device,
    preprocess,
) -> list[list[float]]:
    """Encode frames and predict actions for each consecutive pair. Returns list of 6D actions."""
    feats: list[torch.Tensor] = []
    with torch.no_grad():
        for img in frames:
            x = preprocess(img).unsqueeze(0).to(device)
            z = forward_fn(x)
            feats.append(z.cpu())

    actions: list[list[float]] = []
    with torch.no_grad():
        for i in range(len(feats) - 1):
            z_i    = feats[i].to(device)
            z_next = feats[i + 1].to(device)
            a_norm = idm(z_i, z_next)
            a      = idm.denormalize_action(a_norm)
            actions.append(a[0].tolist())
    return actions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_IDM_CKPT = {
    "mp": "checkpoints/idm_croco_mp_13538757/best.pt",
    "tf": "checkpoints/idm_tf_4hz/epoch_100.pt",
}
_OUTPUT_NAME = {
    "mp": "actions_mp.json",
    "tf": "actions_tf.json",
}


def main() -> None:
    repo = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Predict SE(3) actions from a video or GIF directory.")
    parser.add_argument("--model", choices=["mp", "tf"], required=True,
                        help="IDM variant: 'mp' (mean-pool) or 'tf' (transformer).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",       help="Path to input video file.")
    group.add_argument("--gif_dir",     help="Directory of GIF files to process.")
    group.add_argument("--interp_dir",  help="re10k-mini-interp_py root: scans {clip_id}/{clip_id}_gen.gif.")
    parser.add_argument("--output",      default=None,
                        help="Path to save actions as JSON (default: actions_mp.json / actions_tf.json).")
    parser.add_argument("--idm_ckpt",    default=None,
                        help="Override IDM checkpoint path.")
    parser.add_argument("--croco_ckpt",  default=str(repo / "pretrained_models" / "CroCo.pth"),
                        help="Path to CroCo V2 encoder checkpoint (.pth).")
    parser.add_argument("--every_n",     type=int, default=1,
                        help="Sample every N-th frame for video mode (default 1 = all frames).")
    parser.add_argument("--device",      default=None,
                        help="torch device (default: cuda > cpu).")
    args = parser.parse_args()

    if args.idm_ckpt is None:
        args.idm_ckpt = str(repo / _IDM_CKPT[args.model])

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"device: {device}")

    # -----------------------------------------------------------------------
    # Load CroCo encoder
    # -----------------------------------------------------------------------
    sys.path.insert(0, str(repo))
    from cache_features import PREPROCESS, load_encoder  # noqa: E402

    print(f"loading CroCo encoder from {args.croco_ckpt} ...")
    forward_fn, feat_dim = load_encoder("croco", device, croco_ckpt=Path(args.croco_ckpt))
    print(f"CroCo encoder ready  (feat_dim={feat_dim})")

    # -----------------------------------------------------------------------
    # Load IDM
    # -----------------------------------------------------------------------
    if args.model == "mp":
        from inverse_dynamics import CrocoMeanPoolIDM  # noqa: E402
        idm = CrocoMeanPoolIDM(feat_dim=feat_dim, dropout=0.0)
    else:
        from inverse_dynamics import CrocoTransformerIDM  # noqa: E402
        idm = CrocoTransformerIDM(feat_dim=feat_dim)

    ckpt = torch.load(args.idm_ckpt, map_location=device, weights_only=False)
    idm.load_state_dict(ckpt["model"])
    idm.eval().to(device)
    print(f"IDM ({args.model}) loaded  (epoch={ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.6f})")

    # -----------------------------------------------------------------------
    # interp_dir mode  (re10k-mini-interp_py/{clip_id}/{clip_id}_gen.gif)
    # -----------------------------------------------------------------------
    if args.interp_dir:
        interp_root = Path(args.interp_dir)
        gif_files = sorted(interp_root.glob("*/*_gen.gif"))
        if not gif_files:
            sys.exit(f"[error] no *_gen.gif files found under {interp_root}")
        print(f"\nfound {len(gif_files)} *_gen.gif files under {interp_root}")

        results: dict[str, list[list[float]]] = {}
        for gif_path in gif_files:
            clip_id = gif_path.parent.name
            frames = extract_gif_frames(gif_path)
            if len(frames) < 2:
                print(f"  [skip] {clip_id}: only {len(frames)} frame(s)")
                continue
            actions = predict_actions_for_frames(frames, forward_fn, idm, device, PREPROCESS)
            results[clip_id] = actions
            print(f"  {clip_id}: {len(frames)} frames -> {len(actions)} actions")

        output_path = Path(args.output) if args.output else interp_root / _OUTPUT_NAME[args.model]
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved actions for {len(results)} clips to {output_path}")
        return

    # -----------------------------------------------------------------------
    # GIF directory mode
    # -----------------------------------------------------------------------
    if args.gif_dir:
        gif_dir = Path(args.gif_dir)
        gif_files = sorted(gif_dir.glob("*.gif"))
        if not gif_files:
            sys.exit(f"[error] no GIF files found in {gif_dir}")
        print(f"\nfound {len(gif_files)} GIFs in {gif_dir}")

        results: dict[str, list[list[float]]] = {}
        for gif_path in gif_files:
            frames = extract_gif_frames(gif_path)
            if len(frames) < 2:
                print(f"  [skip] {gif_path.name}: only {len(frames)} frame(s)")
                continue
            actions = predict_actions_for_frames(frames, forward_fn, idm, device, PREPROCESS)
            results[gif_path.name] = actions
            print(f"  {gif_path.name}: {len(frames)} frames -> {len(actions)} actions")

        output_path = Path(args.output) if args.output else gif_dir / _OUTPUT_NAME[args.model]
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved actions for {len(results)} GIFs to {output_path}")
        return

    # -----------------------------------------------------------------------
    # Single video mode
    # -----------------------------------------------------------------------
    print(f"\nextracting frames from {args.video} (every_n={args.every_n}) ...")
    frames = extract_frames(args.video, every_n=args.every_n)
    print(f"extracted {len(frames)} frames")

    if len(frames) < 2:
        sys.exit("[error] need at least 2 frames to predict actions.")

    print("encoding frames ...")
    feats: list[torch.Tensor] = []
    with torch.no_grad():
        for img in frames:
            x = PREPROCESS(img).unsqueeze(0).to(device)
            z = forward_fn(x)
            feats.append(z.cpu())
    print(f"encoded {len(feats)} feature tensors, shape {tuple(feats[0].shape)}")

    print(f"\n{'frame_i':>7}  {'frame_j':>7}  {'vx':>9} {'vy':>9} {'vz':>9}  {'wx':>9} {'wy':>9} {'wz':>9}")
    print("-" * 80)

    with torch.no_grad():
        for i in range(len(feats) - 1):
            z_i    = feats[i].to(device)
            z_next = feats[i + 1].to(device)

            a_norm = idm(z_i, z_next)
            a      = idm.denormalize_action(a_norm)
            v, w   = a[0, :3].tolist(), a[0, 3:].tolist()

            print(f"{i:>7d}  {i+1:>7d}  "
                  f"{v[0]:>9.5f} {v[1]:>9.5f} {v[2]:>9.5f}  "
                  f"{w[0]:>9.5f} {w[1]:>9.5f} {w[2]:>9.5f}")

    print(f"\ndone — predicted {len(feats) - 1} action(s).")


if __name__ == "__main__":
    main()