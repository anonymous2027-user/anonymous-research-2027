"""Participant-disjoint partitions with the exact counts in manuscript Table 1."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from storage import atomic_json, sha256

PARTITIONS = {
    'iid': ((36, 10), (36, 10), (35, 10)),
    'mild': ((36, 7), (36, 10), (35, 13)),
    'severe': ((36, 1), (36, 9), (35, 20)),
    'quantity': ((54, 15), (32, 9), (21, 6)),
}


def validate_windows(frame, split, expected_people=None):
    required = {'participant_id', 'sample_id', 'label', 'split', 'start_time', 'end_time',
                'audio_path', 'transcript_path'}
    if not required <= set(frame):
        raise ValueError(f'Missing manifest fields: {required - set(frame)}')
    if frame.empty or frame.sample_id.duplicated().any() or not frame.label.isin([0, 1]).all():
        raise ValueError('Empty manifest, duplicate sample IDs or invalid labels')
    if not frame.split.eq(split).all() or not (frame.end_time > frame.start_time).all():
        raise ValueError('Wrong split or window boundaries')
    grouped = frame.groupby('participant_id')
    if not grouped.size().eq(200).all() or not grouped.label.nunique().eq(1).all():
        raise ValueError('Expected 200 windows and one label per participant')
    if frame.duplicated(['participant_id', 'start_time', 'end_time']).any():
        raise ValueError('Repeated participant window boundaries')
    if expected_people is not None and grouped.ngroups != expected_people:
        raise ValueError(f'Expected {expected_people} {split} participants')


def assign_participants(frame, partition, seed=42):
    validate_windows(frame, 'train', 107)
    labels = frame.groupby('participant_id', sort=True).label.first()
    if labels.sum() != 30:
        raise ValueError('Expected 30 positive training participants')
    rng = np.random.default_rng(seed)
    positives = rng.permutation(labels[labels == 1].index.to_numpy())
    negatives = rng.permutation(labels[labels == 0].index.to_numpy())
    clients, pos_offset, neg_offset = [], 0, 0
    for total, positive in PARTITIONS[partition]:
        negative = total - positive
        ids = list(positives[pos_offset:pos_offset+positive]) + list(negatives[neg_offset:neg_offset+negative])
        client = frame[frame.participant_id.isin(ids)].copy().reset_index(drop=True)
        validate_windows(client, 'train', total)
        clients.append(client)
        pos_offset += positive
        neg_offset += negative
    if pos_offset != len(positives) or neg_offset != len(negatives):
        raise ValueError('Partition does not exhaust the cohort')
    if set(pd.concat(clients).sample_id) != set(frame.sample_id):
        raise ValueError('Partition does not preserve all windows')
    return clients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.train, dtype={'participant_id': str, 'sample_id': str})
    for name in PARTITIONS:
        folder = args.output/name
        folder.mkdir(parents=True, exist_ok=True)
        for cid, client in enumerate(assign_participants(frame, name, args.seed)):
            client.to_csv(folder/f'client_{cid}.csv', index=False)
        atomic_json(folder/'partition.json', dict(partition=name, seed=args.seed,
                    source_sha256=sha256(args.train), counts=PARTITIONS[name],
                    client_sha256=[sha256(folder/f'client_{cid}.csv') for cid in range(3)]))
        print(name, PARTITIONS[name])


if __name__ == '__main__':
    main()
