"""Window construction helpers for fixed-length interview windows."""

import math
import numpy as np
import pandas as pd

MIN_TARGET_DURATION = 30.0
MAX_TARGET_DURATION = 120.0
CANDIDATE_STEP = 0.5
MIN_PARTICIPANT_SPEECH_RATIO = 0.0

def load_transcript(path: str) -> pd.DataFrame:
    transcript = pd.read_csv(path)
    transcript["speaker"] = (
        transcript["speaker"]
        .astype(str)
        .str.strip()
    )
    transcript["start_time"] = pd.to_numeric(
        transcript["start_time"],
        errors="raise",
    )
    transcript["stop_time"] = pd.to_numeric(
        transcript["stop_time"],
        errors="raise",
    )
    return transcript

def enumerate_reachable_candidates(
    row,
) -> list[dict]:
    pid = int(row.participant_id)
    transcript = load_transcript(
        row.transcript_clean_path
    )
    timestamps = np.load(
        row.timestamps_path,
        mmap_mode="r",
    )
    common_end = min(
        float(row.audio_duration_seconds),
        float(timestamps[-1]),
    )
    ellie_rows = transcript.loc[
        transcript["speaker"] == "Ellie"
    ]
    participant_rows = transcript.loc[
        transcript["speaker"] == "Participant"
    ]
    if ellie_rows.empty:
        raise ValueError(
            f"Participant {pid}: no Ellie utterances"
        )
    if participant_rows.empty:
        raise ValueError(
            f"Participant {pid}: "
            "no Participant utterances"
        )
    # The key is the actual rounded time window.
    # This prevents two transcript-index combinations from
    # producing the same audio/visual segment.
    candidate_map = {}
    for start_index, start_row in (
        ellie_rows.iterrows()
    ):
        original_start = float(
            start_row["start_time"]
        )
        start_time = float(
            math.floor(original_start)
        )
        valid_end_rows = participant_rows.loc[
            (
                participant_rows.index
                > start_index
            )
            & (
                np.ceil(
                    participant_rows["stop_time"]
                )
                <= common_end
            )
        ].copy()
        if valid_end_rows.empty:
            continue
        valid_end_rows = valid_end_rows.sort_values(
            "stop_time"
        )
        original_end_times = (
            valid_end_rows["stop_time"]
            .to_numpy(dtype=np.float64)
        )
        durations = (
            original_end_times - start_time
        )
        end_indices = (
            valid_end_rows.index
            .to_numpy(dtype=np.int64)
        )
        for position in range(len(durations)):
            duration = float(
                durations[position]
            )
            if position == 0:
                left_boundary = -np.inf
            else:
                left_boundary = (
                    durations[position - 1]
                    + duration
                ) / 2.0
            if position == len(durations) - 1:
                right_boundary = np.inf
            else:
                right_boundary = (
                    duration
                    + durations[position + 1]
                ) / 2.0
            interval_start = max(
                MIN_TARGET_DURATION,
                float(left_boundary),
            )
            interval_end = min(
                MAX_TARGET_DURATION,
                float(right_boundary),
            )
            if interval_start > interval_end:
                continue
            # Use the midpoint of the feasible target interval.
            target_duration = (
                interval_start + interval_end
            ) / 2.0
            distances = np.abs(
                durations - target_duration
            )
            selected_position = int(
                np.argmin(distances)
            )
            # Ensure this endpoint is actually selected by
            # the nearest-end rule.
            if selected_position != position:
                continue
            original_end = float(
                original_end_times[position]
            )
            end_time = float(
                math.ceil(original_end)
            )
            if end_time <= start_time:
                continue
            if end_time > common_end:
                continue
            end_index = int(
                end_indices[position]
            )
            key = (
                int(start_time),
                int(end_time),
            )
            candidate = {
                "participant_id": pid,
                "split": "train",
                "label": int(row.label),
                "phq8_score": int(
                    row.phq8_score
                ),
                "start_time": start_time,
                "end_time": end_time,
                "actual_duration": (
                    end_time - start_time
                ),
                "target_duration": (
                    target_duration
                ),
                "original_interviewer_start": (
                    original_start
                ),
                "original_participant_end": (
                    original_end
                ),
                "start_utterance_index": int(
                    start_index
                ),
                "end_utterance_index": (
                    end_index
                ),
                "remove_interviewer_audio": 1,
                "remove_interviewer_visual": 1,
                "keep_interviewer_text": 1,
                "audio_path": row.audio_path,
                "transcript_path": (
                    row.transcript_clean_path
                ),
                "visual_path": row.visual_path,
                "timestamps_path": (
                    row.timestamps_path
                ),
                "valid_mask_path": (
                    row.valid_mask_path
                ),
            }
            # Keep only one representation for each actual
            # rounded time window.
            if key not in candidate_map:
                candidate_map[key] = candidate
    return list(candidate_map.values())

def sample_candidates(
    candidates: list[dict],
    requested_count: int,
    rng: np.random.Generator,
) -> tuple[list[dict], int]:
    candidate_count = len(candidates)
    if candidate_count == 0:
        raise ValueError(
            "No reachable candidate windows"
        )
    if candidate_count >= requested_count:
        selected_indices = rng.choice(
            candidate_count,
            size=requested_count,
            replace=False,
        ).tolist()
        repeated_count = 0
    else:
        unique_indices = rng.permutation(
            candidate_count
        ).tolist()
        deficit = (
            requested_count - candidate_count
        )
        repeated_indices = rng.choice(
            candidate_count,
            size=deficit,
            replace=True,
        ).tolist()
        selected_indices = (
            unique_indices + repeated_indices
        )
        repeated_count = deficit
    rng.shuffle(selected_indices)
    selected = [
        candidates[index].copy()
        for index in selected_indices
    ]
    occurrence_count = {}
    for record in selected:
        key = (
            int(record["start_time"]),
            int(record["end_time"]),
        )
        occurrence = occurrence_count.get(
            key,
            0,
        )
        record["window_occurrence"] = (
            occurrence + 1
        )
        record["is_repeated_window"] = int(
            occurrence > 0
        )
        occurrence_count[key] = (
            occurrence + 1
        )
    return selected, repeated_count

def allocate_counts(counts, total):
    counts = counts.astype(int)
    raw = counts / counts.sum() * total
    alloc = np.floor(raw).astype(int)
    alloc = np.minimum(alloc, counts)
    remain = total - int(alloc.sum())
    frac = raw - np.floor(raw)
    while remain > 0:
        candidates = [i for i in range(len(counts)) if alloc[i] < counts[i]]
        if not candidates:
            raise RuntimeError("Not enough unique windows to allocate.")
        i = max(candidates, key=lambda j: (frac[j], counts[j] - alloc[j]))
        alloc[i] += 1
        frac[i] = -1.0
        remain -= 1
    return alloc

def evenly_select(group, k):
    if k == 0:
        return group.iloc[0:0]
    g = group.sort_values(["midpoint", "start_time", "end_time"]).reset_index(drop=True)
    n = len(g)
    if k > n:
        raise RuntimeError(f"Requested {k} from only {n} candidates")
    idx = np.floor((np.arange(k) + 0.5) * n / k).astype(int)
    idx = np.clip(idx, 0, n - 1)
    if len(np.unique(idx)) != k:
        raise RuntimeError("Even-selection generated duplicate indices.")
    return g.iloc[idx]

def participant_overlap_duration(
    transcript: pd.DataFrame,
    start_time: float,
    end_time: float,
) -> float:
    participant = transcript.loc[
        transcript["speaker"] == "Participant"
    ]
    overlap_start = np.maximum(
        participant["start_time"].to_numpy(
            dtype=np.float64
        ),
        start_time,
    )
    overlap_end = np.minimum(
        participant["stop_time"].to_numpy(
            dtype=np.float64
        ),
        end_time,
    )
    return float(
        np.maximum(
            overlap_end - overlap_start,
            0.0,
        ).sum()
    )

def create_candidate_starts(
    transcript: pd.DataFrame,
    usable_start: float,
    usable_end: float,
    duration: float,
) -> np.ndarray:
    latest_start = usable_end - duration
    if latest_start < usable_start:
        raise ValueError(
            f"Usable duration "
            f"{usable_end - usable_start:.3f}s "
            f"is shorter than {duration}s"
        )
    starts = np.arange(
        usable_start,
        latest_start + 1e-8,
        CANDIDATE_STEP,
        dtype=np.float64,
    )
    valid_starts = []
    for start_time in starts:
        end_time = start_time + duration
        participant_duration = (
            participant_overlap_duration(
                transcript,
                start_time,
                end_time,
            )
        )
        minimum_speech_duration = (
            duration * MIN_PARTICIPANT_SPEECH_RATIO
        )
        if participant_duration > minimum_speech_duration:
            valid_starts.append(start_time)
    return np.asarray(
        valid_starts,
        dtype=np.float64,
    )

def choose_evenly_spaced_starts(
    candidate_starts: np.ndarray,
    number: int,
) -> np.ndarray:
    if len(candidate_starts) < number:
        raise ValueError(
            f"Only {len(candidate_starts)} valid "
            f"candidate windows, but {number} required"
        )
    positions = np.linspace(
        0,
        len(candidate_starts) - 1,
        number,
    )
    indices = np.round(
        positions
    ).astype(np.int64)
    selected = candidate_starts[indices]
    if len(np.unique(selected)) != number:
        raise RuntimeError(
            "Selected window starts are not unique"
        )
    return selected

def find_transcript_indices(
    transcript: pd.DataFrame,
    start_time: float,
    end_time: float,
) -> tuple[int, int]:
    overlap = (
        (transcript["stop_time"] > start_time)
        & (transcript["start_time"] < end_time)
    )
    indices = transcript.index[
        overlap
    ].to_numpy(dtype=np.int64)
    if len(indices) == 0:
        raise RuntimeError(
            f"No transcript rows overlap "
            f"{start_time:.3f}-{end_time:.3f}"
        )
    return int(indices[0]), int(indices[-1])
