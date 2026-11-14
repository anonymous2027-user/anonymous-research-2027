"""Prepare licensed DAIC-WOZ assets; never downloads or redistributes the corpus."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf

from partitions import validate_windows
from storage import atomic_json, sha256
import windows


def read_labels(path, split):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    names = {'Participant_ID': 'participant_id', 'PHQ8_Binary': 'label',
             'PHQ8_Score': 'phq8_score'}
    frame = frame.rename(columns=names)
    if not {'participant_id', 'label', 'phq8_score'} <= set(frame):
        raise ValueError(f'{path}: require Participant_ID, PHQ8_Binary and PHQ8_Score')
    frame = frame[['participant_id', 'label', 'phq8_score']].astype(int)
    if frame.participant_id.duplicated().any() or not frame.label.isin([0, 1]).all():
        raise ValueError('Invalid participant labels')
    frame['split'] = split
    return frame


def select_train(candidates, label, rng):
    sampled, _ = windows.sample_candidates(candidates, 1000 if label else round(1000*30/77), rng)
    frame = pd.DataFrame(sampled).drop_duplicates(['start_time', 'end_time']).copy()
    if len(frame) < 200:
        raise ValueError('Fewer than 200 distinct reachable training windows')
    frame['midpoint'] = (frame.start_time + frame.end_time)/2
    # Stratify by target duration; do not bin by speech length.
    frame['duration_bin'] = pd.cut(frame.target_duration,
        [30, 45, 60, 75, 90, 105, 120.000001], right=False, labels=False)
    if frame.duration_bin.isna().any():
        raise ValueError('Target duration outside training bins')
    counts = frame.groupby('duration_bin').size().reindex(range(6), fill_value=0).to_numpy()
    allocation = windows.allocate_counts(counts, 200)
    selected = pd.concat([windows.evenly_select(frame[frame.duration_bin == b], int(k))
                          for b, k in enumerate(allocation)], ignore_index=True)
    return selected.sort_values(['start_time', 'end_time']).reset_index(drop=True)


def generate(person, processed, rng):
    folder = processed/str(int(person.participant_id))
    metadata = json.loads((folder/'metadata.json').read_text())
    audio = (folder/'audio.wav').resolve()
    if sf.info(audio).samplerate != 16000 or sf.info(audio).channels != 1:
        raise ValueError('Expected original 16 kHz mono audio; do not silently resample')
    transcript_path = (folder/'transcript.csv').resolve()
    timestamps_path = (folder/'visual_timestamps.npy').resolve()
    common_end = min(float(metadata['audio_duration_seconds']), float(np.load(timestamps_path)[-1]))
    if person.split == 'train':
        row = SimpleNamespace(**person._asdict(), transcript_clean_path=str(transcript_path),
            timestamps_path=str(timestamps_path), audio_duration_seconds=metadata['audio_duration_seconds'],
            audio_path=str(audio), visual_path=str(folder/'visual_30fps.npy'),
            valid_mask_path=str(folder/'visual_valid_mask.npy'))
        selected = select_train(windows.enumerate_reachable_candidates(row), person.label, rng)
    else:
        transcript = windows.load_transcript(transcript_path)
        usable_start = max(0.0, float(transcript.start_time.min()))
        usable_end = min(common_end, float(transcript.stop_time.max()))
        starts = windows.create_candidate_starts(transcript, usable_start, usable_end, 30.0)
        starts = windows.choose_evenly_spaced_starts(starts, 200)
        selected = pd.DataFrame({'start_time': starts, 'end_time': starts + 30.0})
    # Only audio and text enter the model; visual timestamps only bound the window range.
    selected['participant_id'] = str(int(person.participant_id))
    selected['label'] = int(person.label)
    selected['split'] = person.split
    selected['audio_path'] = str(audio)
    selected['transcript_path'] = str(transcript_path)
    selected['sample_id'] = [f'{person.split}_{int(person.participant_id)}_{i:03d}' for i in range(200)]
    return selected[['sample_id', 'participant_id', 'label', 'split', 'start_time', 'end_time',
                     'audio_path', 'transcript_path']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-labels', required=True, type=Path)
    parser.add_argument('--dev-labels', required=True, type=Path)
    parser.add_argument('--test-labels', required=True, type=Path)
    parser.add_argument('--raw-root', type=Path, help='Features directory with Audio, Transcript, AUs, gaze, pose')
    parser.add_argument('--processed', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--skip-clean', action='store_true', help='Reuse participant-level cleaned assets')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    labels = {split: read_labels(path, split) for split, path in
              [('train', args.train_labels), ('dev', args.dev_labels), ('test', args.test_labels)]}
    cohorts = [set(frame.participant_id) for frame in labels.values()]
    if any(cohorts[i] & cohorts[j] for i in range(3) for j in range(i)):
        raise ValueError('Participant leakage across official splits')
    for (split, frame), expected, positives in zip(labels.items(), [107, 35, 47], [30, 12, 14]):
        if len(frame) != expected or frame.label.sum() != positives:
            raise ValueError(f'Wrong official {split} cohort')
    if not args.skip_clean:
        if args.raw_root is None:
            parser.error('--raw-root is required unless --skip-clean')
        import visual_clean
        visual_clean.RAW_ROOT = args.raw_root.resolve()
        visual_clean.PROCESSED_ROOT = args.processed.resolve()
        for pid in sorted(set.union(*cohorts)):
            visual_clean.parse_args = lambda pid=pid: SimpleNamespace(pid=int(pid))
            visual_clean.main()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for split, cohort in labels.items():
        result = pd.concat([generate(person, args.processed.resolve(), rng)
                            for person in cohort.itertuples(index=False)], ignore_index=True)
        validate_windows(result, split, len(cohort))
        result.to_csv(args.output/f'{split}.csv', index=False)
        print(split, len(cohort), len(result), flush=True)
    atomic_json(args.output/'preprocessing.json', dict(seed=args.seed,
        protocol='interview-span windows; participant audio concatenation; first 30 s model cap',
        labels_sha256={s: sha256(p) for s, p in [('train', args.train_labels),
            ('dev', args.dev_labels), ('test', args.test_labels)]},
        manifests_sha256={s: sha256(args.output/f'{s}.csv') for s in labels}))


if __name__ == '__main__':
    main()
