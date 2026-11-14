from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import soundfile as sf
RAW_ROOT = Path("data/raw/Features")
PROCESSED_ROOT = Path("data/processed/participant_level")
AU_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r",
    "AU06_r", "AU09_r", "AU10_r", "AU12_r",
    "AU14_r", "AU15_r", "AU17_r", "AU20_r",
    "AU25_r", "AU26_r",
]
GAZE_COLUMNS = [
    "x_0", "y_0", "z_0",
    "x_1", "y_1", "z_1",
    "x_h0", "y_h0", "z_h0",
    "x_h1", "y_h1", "z_h1",
]
POSE_COLUMNS = [
    "Tx", "Ty", "Tz",
    "Rx", "Ry", "Rz",
]
FEATURE_COLUMNS = AU_COLUMNS + GAZE_COLUMNS + POSE_COLUMNS
def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean one DAIC-WoZ participant."
    )
    parser.add_argument(
        "--pid",
        type=int,
        required=True,
        help="Participant ID, for example 300",
    )
    return parser.parse_args()
def read_visual(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(
        path,
        skipinitialspace=True,
    )
    df.columns = df.columns.str.strip()
    return df
def check_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    file_name: str,
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{file_name} is missing columns: {missing}"
        )
def main() -> None:
    args = parse_args()
    pid = args.pid
    output_root = PROCESSED_ROOT / str(pid)
    audio_path = (
        RAW_ROOT / "Audio" / f"{pid}_AUDIO.wav"
    )
    transcript_path = (
        RAW_ROOT / "Transcript" / f"{pid}_TRANSCRIPT.csv"
    )
    aus_path = (
        RAW_ROOT / "AUs" / f"{pid}_CLNF_AUs.txt"
    )
    gaze_path = (
        RAW_ROOT / "gaze" / f"{pid}_CLNF_gaze.txt"
    )
    pose_path = (
        RAW_ROOT / "pose" / f"{pid}_CLNF_pose.txt"
    )
    required_files = [
        audio_path,
        transcript_path,
        aus_path,
        gaze_path,
        pose_path,
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    # Read raw data
    aus = read_visual(aus_path)
    gaze = read_visual(gaze_path)
    pose = read_visual(pose_path)
    transcript = pd.read_csv(
        transcript_path,
        sep="\t",
    )
    transcript.columns = transcript.columns.str.strip()
    audio_info = sf.info(audio_path)
    audio_duration = (
        audio_info.frames / audio_info.samplerate
    )
    # Check columns
    metadata_columns = [
        "frame",
        "timestamp",
        "confidence",
        "success",
    ]
    check_required_columns(
        aus,
        metadata_columns + AU_COLUMNS,
        aus_path.name,
    )
    check_required_columns(
        gaze,
        metadata_columns + GAZE_COLUMNS,
        gaze_path.name,
    )
    check_required_columns(
        pose,
        metadata_columns + POSE_COLUMNS,
        pose_path.name,
    )
    check_required_columns(
        transcript,
        [
            "start_time",
            "stop_time",
            "speaker",
            "value",
        ],
        transcript_path.name,
    )
    # Check visual alignment
    if not (
        len(aus) == len(gaze) == len(pose)
    ):
        raise ValueError(
            "AUs, gaze and pose have different row counts."
        )
    aus_frames = aus["frame"].to_numpy()
    gaze_frames = gaze["frame"].to_numpy()
    pose_frames = pose["frame"].to_numpy()
    if not np.array_equal(
        aus_frames,
        gaze_frames,
    ):
        raise ValueError(
            "AUs and gaze frame indices do not match."
        )
    if not np.array_equal(
        aus_frames,
        pose_frames,
    ):
        raise ValueError(
            "AUs and pose frame indices do not match."
        )
    timestamps = aus["timestamp"].to_numpy(
        dtype=np.float64
    )
    if not np.allclose(
        timestamps,
        gaze["timestamp"].to_numpy(),
        atol=1e-5,
    ):
        raise ValueError(
            "AUs and gaze timestamps do not match."
        )
    if not np.allclose(
        timestamps,
        pose["timestamp"].to_numpy(),
        atol=1e-5,
    ):
        raise ValueError(
            "AUs and pose timestamps do not match."
        )
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            "Visual timestamps are not strictly increasing."
        )
    # Build the 32-dimensional feature sequence
    visual = pd.concat(
        [
            aus[AU_COLUMNS],
            gaze[GAZE_COLUMNS],
            pose[POSE_COLUMNS],
        ],
        axis=1,
    ).astype(np.float32)
    # A frame is valid only if all feature sources are reliable
    valid_mask = (
        (aus["success"].to_numpy() == 1)
        & (gaze["success"].to_numpy() == 1)
        & (pose["success"].to_numpy() == 1)
        & (aus["confidence"].to_numpy() >= 0.8)
        & (gaze["confidence"].to_numpy() >= 0.8)
        & (pose["confidence"].to_numpy() >= 0.8)
    )
    invalid_count = int((~valid_mask).sum())
    # Preserve every frame but mark unreliable data as missing
    visual.loc[~valid_mask, :] = np.nan
    # Handle sentinel values such as -100
    visual = visual.mask(visual <= -99)
    # Interpolate along time without deleting frames
    visual = visual.interpolate(
        method="linear",
        axis=0,
        limit_direction="both",
    )
    if visual.isna().any().any():
        missing_columns = visual.columns[
            visual.isna().any()
        ].tolist()
        raise ValueError(
            "NaN values remain after interpolation in "
            f"columns: {missing_columns}"
        )
    visual_array = visual.to_numpy(
        dtype=np.float32
    )
    # Save arrays
    np.save(
        output_root / "visual_30fps.npy",
        visual_array,
    )
    np.save(
        output_root / "visual_timestamps.npy",
        timestamps,
    )
    np.save(
        output_root / "visual_valid_mask.npy",
        valid_mask.astype(np.bool_),
    )
    transcript.to_csv(
        output_root / "transcript.csv",
        index=False,
    )
    # Create a symbolic link rather than duplicate the WAV file
    audio_link = output_root / "audio.wav"
    if audio_link.exists() or audio_link.is_symlink():
        audio_link.unlink()
    audio_link.symlink_to(
        audio_path.resolve()
    )
    estimated_fps = float(
        (len(timestamps) - 1)
        / (timestamps[-1] - timestamps[0])
    )
    metadata = {
        "participant_id": pid,
        "audio_path": str(audio_path.resolve()),
        "audio_sample_rate": int(
            audio_info.samplerate
        ),
        "audio_channels": int(
            audio_info.channels
        ),
        "audio_duration_seconds": float(
            audio_duration
        ),
        "visual_num_frames": int(
            visual_array.shape[0]
        ),
        "visual_dim": int(
            visual_array.shape[1]
        ),
        "visual_first_timestamp": float(
            timestamps[0]
        ),
        "visual_last_timestamp": float(
            timestamps[-1]
        ),
        "visual_estimated_fps": estimated_fps,
        "visual_invalid_frames": invalid_count,
        "visual_valid_frames": int(
            valid_mask.sum()
        ),
        "visual_valid_ratio": float(
            valid_mask.mean()
        ),
        "transcript_num_utterances": int(
            len(transcript)
        ),
        "transcript_first_time": float(
            transcript["start_time"].min()
        ),
        "transcript_last_time": float(
            transcript["stop_time"].max()
        ),
        "feature_columns": FEATURE_COLUMNS,
    }
    with open(
        output_root / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )
    print("========== CLEANING COMPLETE ==========")
    print("Participant:", pid)
    print("Visual shape:", visual_array.shape)
    print(
        "Expected visual dimension:",
        len(FEATURE_COLUMNS),
    )
    print("Valid frames:", int(valid_mask.sum()))
    print("Invalid frames:", invalid_count)
    print(
        "Valid ratio:",
        f"{valid_mask.mean():.4%}",
    )
    print(
        "NaN remaining:",
        int(np.isnan(visual_array).sum()),
    )
    print("Estimated FPS:", estimated_fps)
    print("Output directory:", output_root)
    print("\nSaved files:")
    for path in sorted(output_root.iterdir()):
        print(" ", path.name)
