"""
Compare mp and tf predicted actions against gt_relative_actions.json.
Also accumulates predicted action sequences into a final SE(3) pose and
compares translation/rotation error against the GT final pose from gt_action.json.
Only clips present in all files are evaluated.

Usage:
    python compare_actions.py
    python compare_actions.py --mp actions_mp.json --tf actions_tf.json \
        --gt gt_relative_actions.json --gt_pose gt_action.json \
        --output action_comparison.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPONENT_LABELS = ["vx", "vy", "vz", "wx", "wy", "wz"]
COLORS = {"mp": "steelblue", "tf": "tomato"}


# ---------------------------------------------------------------------------
# SE(3) helpers (numpy)
# ---------------------------------------------------------------------------

def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=w.dtype)


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """(6,) twist (v, omega) -> (4, 4) homogeneous SE(3) matrix."""
    v, w = xi[:3], xi[3:]
    theta = np.linalg.norm(w)
    W = _skew(w)
    I3 = np.eye(3, dtype=xi.dtype)
    if theta < 1e-4:
        a = 1.0 - theta**2 / 6.0
        b = 0.5  - theta**2 / 24.0
        c = 1.0 / 6.0 - theta**2 / 120.0
    else:
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta**2
        c = (theta - math.sin(theta)) / theta**3
    R = I3 + a * W + b * (W @ W)
    V = I3 + b * W + c * (W @ W)
    T = np.eye(4, dtype=xi.dtype)
    T[:3, :3] = R
    T[:3,  3] = V @ v
    return T


def compose_twists(actions: np.ndarray) -> np.ndarray:
    """(n, 6) twists -> (4, 4) cumulative SE(3) transform (E_n @ E_0^{-1}).

    Right-to-left composition matches dataset.py's relative_pose convention:
    xi_t = log(E_{t+1} @ E_t^{-1}), so composing gives E_n @ E_0^{-1}.
    """
    Ts = [exp_se3(a) for a in actions]
    out = Ts[-1]
    for i in range(len(Ts) - 2, -1, -1):
        out = out @ Ts[i]
    return out


def se3_inv(T: np.ndarray) -> np.ndarray:
    """Analytic inverse for a (4, 4) SE(3) matrix."""
    R, t = T[:3, :3], T[:3, 3]
    Rt = R.T
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = Rt
    T_inv[:3,  3] = -Rt @ t
    return T_inv


def pose_error(T_pred: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """Translation L2 and rotation angle (degrees) between two (4,4) transforms."""
    t_err = float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]))
    R_diff = T_pred[:3, :3] @ T_gt[:3, :3].T
    cos_t = np.clip((np.trace(R_diff) - 1.0) * 0.5, -1.0, 1.0)
    r_err = float(math.degrees(math.acos(cos_t)))
    return t_err, r_err


def raw_pose_to_mat(raw_pose: list[float]) -> np.ndarray:
    """18-element raw_pose row -> (4, 4) homogeneous extrinsic E.

    Layout: [fx, fy, cx, cy, 0, 0, R[0,0], R[0,1], R[0,2], t[0],
             R[1,0], R[1,1], R[1,2], t[1], R[2,0], R[2,1], R[2,2], t[2]]
    Elements [6:18] are the 3x4 [R|t] matrix in row-major order.
    """
    P = np.array(raw_pose[6:], dtype=np.float64).reshape(3, 4)
    E = np.eye(4, dtype=np.float64)
    E[:3, :] = P
    return E


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_actions(path: str) -> dict[str, np.ndarray]:
    """Load {clip_id: [[float]*6, ...]} -> {clip_id: (T, 6) array}."""
    with open(path) as f:
        raw = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


def load_gt_pose(path: str) -> dict[str, dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_action_metrics(pred: dict, gt: dict, common: list[str]):
    pred_all = np.concatenate([pred[k] for k in common], axis=0)
    gt_all   = np.concatenate([gt[k]   for k in common], axis=0)
    sq_err = (pred_all - gt_all) ** 2
    per_component_mse = sq_err.mean(axis=0)
    total_mse = sq_err.mean()
    per_clip_mse = np.array([((pred[k] - gt[k]) ** 2).mean() for k in common])
    return pred_all, gt_all, per_component_mse, total_mse, per_clip_mse


def compute_final_pose_errors(
    pred: dict[str, np.ndarray],
    gt_pose: dict[str, dict],
    common: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate predicted twists and compare final pose to GT.

    Returns:
        trans_errs: (N,) translation L2 errors
        rot_errs:   (N,) rotation errors in degrees
    """
    trans_errs, rot_errs = [], []
    for clip_id in common:
        raw_poses = gt_pose[clip_id]["raw_poses"]
        E_first = raw_pose_to_mat(raw_poses[0])
        E_last  = raw_pose_to_mat(raw_poses[-1])
        T_gt = E_last @ se3_inv(E_first)

        actions = pred[clip_id].astype(np.float64)
        T_pred = compose_twists(actions)

        t_err, r_err = pose_error(T_pred, T_gt)
        trans_errs.append(t_err)
        rot_errs.append(r_err)
    return np.array(trans_errs), np.array(rot_errs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--mp",      default=str(repo / "results" / "actions_mp.json"))
    parser.add_argument("--tf",      default=str(repo / "results" / "actions_tf.json"))
    parser.add_argument("--gt",      default=str(repo / "results" / "gt_relative_actions.json"))
    parser.add_argument("--gt_pose", default=str(repo / "results" / "gt_action.json"))
    parser.add_argument("--output",  default=str(repo / "results" / "action_comparison.png"))
    args = parser.parse_args()

    mp_pred  = load_actions(args.mp)
    tf_pred  = load_actions(args.tf)
    gt       = load_actions(args.gt)
    gt_pose  = load_gt_pose(args.gt_pose)

    common = sorted(set(mp_pred) & set(tf_pred) & set(gt) & set(gt_pose))
    if not common:
        raise ValueError("No clip IDs common to all four files.")
    print(f"mp: {len(mp_pred)},  tf: {len(tf_pred)},  gt: {len(gt)},  gt_pose: {len(gt_pose)}")
    print(f"Common to all: {len(common)} clips\n")

    # Action MSE
    mp_all, gt_all, mp_comp_mse, mp_total, mp_clip_mse = compute_action_metrics(mp_pred, gt, common)
    tf_all, _,      tf_comp_mse, tf_total, tf_clip_mse = compute_action_metrics(tf_pred, gt, common)

    # Final pose error
    mp_trans, mp_rot = compute_final_pose_errors(mp_pred, gt_pose, common)
    tf_trans, tf_rot = compute_final_pose_errors(tf_pred, gt_pose, common)

    # Print table
    print(f"{'':6s}  {'mp MSE':>10s}  {'tf MSE':>10s}")
    print("-" * 32)
    for label, mp_v, tf_v in zip(COMPONENT_LABELS, mp_comp_mse, tf_comp_mse):
        print(f"{label:6s}  {mp_v:10.6f}  {tf_v:10.6f}")
    print("-" * 32)
    print(f"{'total':6s}  {mp_total:10.6f}  {tf_total:10.6f}")
    print()
    print(f"Final pose error (mean over {len(common)} clips):")
    print(f"  Translation — mp: {mp_trans.mean():.5f},  tf: {tf_trans.mean():.5f}")
    print(f"  Rotation    — mp: {mp_rot.mean():.3f}°,  tf: {tf_rot.mean():.3f}°")

    # -----------------------------------------------------------------------
    # Plot: 2 rows x 3 panels
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"MP vs TF  |  action MSE — mp: {mp_total:.5f}, tf: {tf_total:.5f}"
        f"  |  N = {len(common)} clips",
        fontsize=13,
    )

    # Row 0 — Action MSE
    # 0,0: per-component MSE grouped bars
    ax = axes[0, 0]
    x = np.arange(len(COMPONENT_LABELS))
    w = 0.35
    b1 = ax.bar(x - w / 2, mp_comp_mse, w, label="mp", color=COLORS["mp"])
    b2 = ax.bar(x + w / 2, tf_comp_mse, w, label="tf", color=COLORS["tf"])
    ax.set_xticks(x); ax.set_xticklabels(COMPONENT_LABELS)
    ax.set_title("Per-component MSE"); ax.set_ylabel("MSE"); ax.legend()
    for bar, v in list(zip(b1, mp_comp_mse)) + list(zip(b2, tf_comp_mse)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)

    # 0,1: per-clip MSE histogram
    ax = axes[0, 1]
    bins = np.linspace(0, max(mp_clip_mse.max(), tf_clip_mse.max()), 31)
    ax.hist(mp_clip_mse, bins=bins, alpha=0.6, color=COLORS["mp"], label=f"mp (mean={mp_clip_mse.mean():.5f})")
    ax.hist(tf_clip_mse, bins=bins, alpha=0.6, color=COLORS["tf"], label=f"tf (mean={tf_clip_mse.mean():.5f})")
    ax.axvline(mp_clip_mse.mean(), color=COLORS["mp"], linestyle="--", linewidth=1)
    ax.axvline(tf_clip_mse.mean(), color=COLORS["tf"], linestyle="--", linewidth=1)
    ax.set_title("Per-clip MSE distribution"); ax.set_xlabel("MSE"); ax.set_ylabel("Count"); ax.legend(fontsize=8)

    # 0,2: predicted vs GT scatter
    ax = axes[0, 2]
    comp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (label, cc) in enumerate(zip(COMPONENT_LABELS, comp_colors)):
        ax.scatter(gt_all[:, i], mp_all[:, i], s=2, alpha=0.25, color=cc, marker="o")
        ax.scatter(gt_all[:, i], tf_all[:, i], s=2, alpha=0.25, color=cc, marker="^")
    lim_min = min(gt_all.min(), mp_all.min(), tf_all.min())
    lim_max = max(gt_all.max(), mp_all.max(), tf_all.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="perfect")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", label="mp"),
               Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", label="tf")]
    handles += [Line2D([0], [0], marker="s", color="w", markerfacecolor=cc, label=lbl)
                for lbl, cc in zip(COMPONENT_LABELS, comp_colors)]
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.set_title("Predicted vs GT (all components)"); ax.set_xlabel("GT"); ax.set_ylabel("Predicted")

    # Row 1 — Final pose error
    # 1,0: translation error distribution
    ax = axes[1, 0]
    t_max = max(mp_trans.max(), tf_trans.max())
    bins_t = np.linspace(0, t_max, 31)
    ax.hist(mp_trans, bins=bins_t, alpha=0.6, color=COLORS["mp"], label=f"mp (mean={mp_trans.mean():.4f})")
    ax.hist(tf_trans, bins=bins_t, alpha=0.6, color=COLORS["tf"], label=f"tf (mean={tf_trans.mean():.4f})")
    ax.axvline(mp_trans.mean(), color=COLORS["mp"], linestyle="--", linewidth=1)
    ax.axvline(tf_trans.mean(), color=COLORS["tf"], linestyle="--", linewidth=1)
    ax.set_title("Final pose — translation error (L2)"); ax.set_xlabel("Translation error"); ax.set_ylabel("Count"); ax.legend(fontsize=8)

    # 1,1: rotation error distribution
    ax = axes[1, 1]
    r_max = max(mp_rot.max(), tf_rot.max())
    bins_r = np.linspace(0, r_max, 31)
    ax.hist(mp_rot, bins=bins_r, alpha=0.6, color=COLORS["mp"], label=f"mp (mean={mp_rot.mean():.3f}°)")
    ax.hist(tf_rot, bins=bins_r, alpha=0.6, color=COLORS["tf"], label=f"tf (mean={tf_rot.mean():.3f}°)")
    ax.axvline(mp_rot.mean(), color=COLORS["mp"], linestyle="--", linewidth=1)
    ax.axvline(tf_rot.mean(), color=COLORS["tf"], linestyle="--", linewidth=1)
    ax.set_title("Final pose — rotation error (deg)"); ax.set_xlabel("Rotation error (°)"); ax.set_ylabel("Count"); ax.legend(fontsize=8)

    # 1,2: per-clip scatter: mp trans err vs tf trans err
    ax = axes[1, 2]
    ax.scatter(mp_trans, tf_trans, s=8, alpha=0.5, color="purple")
    lim = max(mp_trans.max(), tf_trans.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="equal")
    ax.set_title("Translation error: mp vs tf (per clip)")
    ax.set_xlabel("mp error"); ax.set_ylabel("tf error"); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
