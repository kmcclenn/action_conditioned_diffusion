"""
Download RealEstate10K frames from YouTube.

For each clip, fetches a direct stream URL via yt-dlp and runs a single
ffmpeg pass over the clip's time window. Frame PTS values are parsed from
ffmpeg's showinfo filter output and matched to the dataset's timestamps.

Output layout:
    {frames_root}/{split}/{clip_id}/{timestamp_us}.jpg

Usage:
    python download_frames.py RealEstate10K/ frames/ --split train --workers 8
    python download_frames.py RealEstate10K/ frames/ --split test --limit 100
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dataset import parse_clip


def _yt_stream_url(youtube_url: str, timeout: int = 30) -> str:
    """Return the best direct video-stream URL via yt-dlp."""
    result = subprocess.run(
        [
            "yt-dlp", "-g",
            "--format", "bestvideo[ext=mp4]/best[ext=mp4]/best",
            youtube_url,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "yt-dlp returned non-zero")
    # yt-dlp may return two lines (video+audio DASH); take the video line
    return result.stdout.strip().splitlines()[0]


def _extract_frames(
    stream_url: str,
    timestamps_us: list[int],
    out_dir: Path,
    img_size: tuple[int, int] | None,
    quality: int,
) -> int:
    """
    Extract frames at the requested microsecond timestamps from a stream.

    Runs a single ffmpeg pass over [start-1s, end+1s], captures per-frame
    PTS from the showinfo filter, then copies each best-matching frame to
    out_dir/{timestamp_us}.jpg.

    Returns the number of frames saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    start_s = timestamps_us[0] / 1e6
    end_s   = timestamps_us[-1] / 1e6
    seek_s  = max(0.0, start_s - 1.0)
    to_s    = end_s + 1.0

    vf_parts = ["showinfo"]
    if img_size:
        vf_parts.append(f"scale={img_size[0]}:{img_size[1]}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frame_pat = str(tmp / "%06d.jpg")

        proc = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{seek_s:.3f}",
                "-to", f"{to_s:.3f}",
                "-i", stream_url,
                "-vf", ",".join(vf_parts),
                "-q:v", str(quality),
                "-vsync", "0",
                frame_pat,
            ],
            capture_output=True, text=True,
        )

        # showinfo lines: "... n:  42 pts: 12345 pts_time:1.234 ..."
        pts_times: list[float] = [
            float(m.group(1))
            for line in proc.stderr.splitlines()
            if (m := re.search(r"pts_time:(\S+)", line))
        ]

        extracted = sorted(tmp.glob("*.jpg"))
        if not extracted:
            raise RuntimeError(
                f"ffmpeg produced no frames.\nstderr: {proc.stderr[-1000:]}"
            )

        # If showinfo count mismatches (rare), fall back to uniform spacing
        if len(pts_times) != len(extracted):
            if pts_times:
                interval = (end_s - start_s) / max(len(extracted) - 1, 1)
                pts_times = [pts_times[0] + i * interval for i in range(len(extracted))]
            else:
                interval = (end_s - start_s) / max(len(extracted) - 1, 1)
                pts_times = [start_s + i * interval for i in range(len(extracted))]

        saved = 0
        for ts_us in timestamps_us:
            ts_s = ts_us / 1e6
            best = min(range(len(pts_times)), key=lambda i: abs(pts_times[i] - ts_s))
            dst = out_dir / f"{ts_us}.jpg"
            shutil.copy2(extracted[best], dst)
            saved += 1

        return saved


def download_clip(
    clip_path: Path,
    frames_root: Path,
    img_size: tuple[int, int] | None = None,
    quality: int = 2,
    skip_existing: bool = True,
) -> tuple[str, bool, str]:
    """
    Download all frames for one clip. Safe to call from a subprocess worker.

    Returns (clip_id, success, message).
    """
    clip_id = clip_path.stem
    split   = clip_path.parent.name
    out_dir = frames_root / split / clip_id

    clip = parse_clip(clip_path)
    timestamps_us: list[int] = clip["timestamps"].tolist()

    if skip_existing:
        have = {p.stem for p in out_dir.glob("*.jpg")}
        timestamps_us = [ts for ts in timestamps_us if str(ts) not in have]
        if not timestamps_us:
            return clip_id, True, "already complete"

    try:
        url = _yt_stream_url(clip["url"])
    except Exception as exc:
        return clip_id, False, f"yt-dlp: {exc}"

    try:
        n = _extract_frames(url, timestamps_us, out_dir, img_size, quality)
        return clip_id, True, f"{n} frames saved"
    except Exception as exc:
        return clip_id, False, f"ffmpeg: {exc}"


def _worker(args: tuple) -> tuple[str, bool, str]:
    clip_path, frames_root, img_size, quality, skip_existing = args
    return download_clip(clip_path, frames_root, img_size, quality, skip_existing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RealEstate10K frames")
    parser.add_argument("dataset_root", help="Path to RealEstate10K/ directory")
    parser.add_argument("frames_root",  help="Output directory for downloaded frames")
    parser.add_argument("--split",    default="train", choices=["train", "test"])
    parser.add_argument("--workers",  type=int, default=4,
                        help="Parallel download workers (each runs yt-dlp + ffmpeg)")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Process at most this many clips (useful for testing)")
    parser.add_argument("--img_size", type=str, default=None,
                        help="Resize frames to WxH, e.g. 256x256")
    parser.add_argument("--quality",  type=int, default=2,
                        help="JPEG quality (ffmpeg -q:v, lower=better, 1-31)")
    parser.add_argument("--no_skip",  action="store_true",
                        help="Re-download frames that already exist on disk")
    args = parser.parse_args()

    img_size: tuple[int, int] | None = None
    if args.img_size:
        w, h = args.img_size.lower().split("x")
        img_size = (int(w), int(h))

    clip_dir = Path(args.dataset_root) / args.split
    clips = sorted(clip_dir.glob("*.txt"))
    if not clips:
        raise SystemExit(f"No .txt files found in {clip_dir}")

    # Filter to the canonical shared subset if present. Keeps collaborators on
    # the same clips without any CLI dance.
    clip_list_path = Path(__file__).parent / "clip_lists" / f"{args.split}.txt"
    if clip_list_path.exists():
        wanted = set(clip_list_path.read_text().split())
        clips = [c for c in clips if c.stem in wanted]
        missing = wanted - {c.stem for c in clips}
        if missing:
            raise SystemExit(
                f"{len(missing)} clip IDs in {clip_list_path} are not in {clip_dir} "
                f"(e.g. {sorted(missing)[:3]})."
            )
        print(f"Restricted to {len(clips)} clips from {clip_list_path}")

    if args.limit:
        clips = clips[: args.limit]

    frames_root = Path(args.frames_root)
    skip = not args.no_skip

    print(f"Downloading {len(clips)} clips ({args.split}) → {frames_root}  workers={args.workers}")

    tasks = [(c, frames_root, img_size, args.quality, skip) for c in clips]

    ok = fail = skip_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, t): t[0].stem for t in tasks}
        for fut in as_completed(futs):
            clip_id, success, msg = fut.result()
            if success:
                if msg == "already complete":
                    skip_count += 1
                else:
                    ok += 1
            else:
                fail += 1
            status = "OK  " if success else "FAIL"
            print(f"  [{status}] {clip_id}: {msg}")

    print(f"\nDone. ok={ok}  skipped={skip_count}  failed={fail}")


if __name__ == "__main__":
    main()
