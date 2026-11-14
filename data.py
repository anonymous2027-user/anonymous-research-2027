"""Participant-only waveform and full conversational text for a manifest window."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from partitions import validate_windows


def load_manifest(path, split, expected_people=None):
    frame = pd.read_csv(path, dtype={'participant_id': str, 'sample_id': str})
    validate_windows(frame, split, expected_people)
    return frame


def asset_path(value, root):
    path = Path(value)
    return path if path.is_absolute() else Path(root)/path


@lru_cache(maxsize=256)
def transcript(path):
    frame = pd.read_csv(path)
    if not {'start_time', 'stop_time', 'speaker', 'value'} <= set(frame):
        raise ValueError('Clean transcript must have start_time, stop_time, speaker, value')
    frame = frame.copy()
    frame['speaker'] = frame.speaker.astype(str).str.strip()
    frame['value'] = frame.value.fillna('').astype(str)
    if not frame.speaker.str.lower().isin(['ellie', 'interviewer', 'doctor', 'clinician', 'participant']).all():
        raise ValueError('Unexpected speaker in cleaned transcript')
    return frame.sort_values(['start_time', 'stop_time'], kind='stable')


class AudioTextDataset:
    def __init__(self, frame, data_root='.'):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(data_root)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        text = transcript(str(asset_path(row.transcript_path, self.root).resolve()))
        turns = text[(text.stop_time > row.start_time) & (text.start_time < row.end_time)]
        interviewer = {'ellie', 'interviewer', 'doctor', 'clinician'}
        conversation = '\n'.join(
            ('Interviewer' if r.speaker.lower() in interviewer else 'Participant')+': '+r.value.strip()
            for r in turns.itertuples(index=False) if r.value.strip())
        intervals = []
        for turn in turns[~turns.speaker.str.lower().isin(interviewer)].itertuples(index=False):
            start, end = max(float(turn.start_time), row.start_time), min(float(turn.stop_time), row.end_time)
            if end <= start:
                continue
            if intervals and start <= intervals[-1][1]+0.02:
                intervals[-1][1] = max(intervals[-1][1], end)
            else:
                intervals.append([start, end])
        pieces = []
        with sf.SoundFile(asset_path(row.audio_path, self.root)) as stream:
            if stream.samplerate != 16000 or stream.channels != 1:
                raise ValueError('Expected 16 kHz mono audio')
            for start, end in intervals:
                first, last = max(0, round(start*16000)), min(len(stream), round(end*16000))
                if last <= first:
                    continue
                stream.seek(first)
                pieces.append(stream.read(last-first, dtype='float32'))
        if not pieces or not sum(map(len, pieces)) or not conversation:
            raise ValueError(f'Empty speech or transcript: {row.sample_id}')
        audio = np.concatenate(pieces)
        if not np.isfinite(audio).all():
            raise ValueError('Nonfinite waveform')
        return dict(audio=audio, conversation_text=conversation,
                    target_text='Depressed' if row.label else 'Not Depressed',
                    participant_id=str(row.participant_id), sample_id=str(row.sample_id), label=int(row.label))


def verify_assets(frames, root):
    """Content identities bind resumable runs to the licensed assets, without copying them."""
    from storage import sha256
    paths = set()
    for frame in frames:
        for column in ['audio_path', 'transcript_path']:
            paths.update(str(asset_path(value, root).resolve()) for value in frame[column].unique())
    return sorted(sha256(path) for path in paths)
