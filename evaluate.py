"""Evaluate each adapter: freeze its Dev-only threshold before accessing Test scores."""
import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import pandas as pd

from config import Settings
from data import AudioTextDataset, load_manifest, verify_assets
from metrics import summarize_scores
from storage import atomic_json, sha256
from train import run_lock


def read_scores(path):
    return pd.read_csv(path, dtype={'sample_id': str, 'participant_id': str})


def score_windows(backend, manifest, data_root, path, deadline):
    fields = ['sample_id', 'participant_id', 'true_label', 'nll_margin']
    completed = set()
    if path.exists() and path.stat().st_size:
        existing = read_scores(path)
        if not set(fields) <= set(existing) or existing.sample_id.duplicated().any():
            raise ValueError('Invalid partial score file')
        if not set(existing.sample_id) <= set(manifest.sample_id):
            raise ValueError('Unknown sample in partial score file')
        aligned = manifest.set_index('sample_id').loc[existing.sample_id]
        if (existing.participant_id.to_numpy() != aligned.participant_id.to_numpy()).any() or (
                existing.true_label.to_numpy() != aligned.label.to_numpy()).any():
            raise ValueError('Partial scores have wrong participant or label')
        if not all(math.isfinite(v) for v in existing.nll_margin):
            raise ValueError('Nonfinite partial scores')
        completed = set(existing.sample_id)
    dataset = AudioTextDataset(manifest, data_root)
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        for index, row in enumerate(manifest.itertuples(index=False)):
            if row.sample_id in completed:
                continue
            if time.monotonic() >= deadline:
                break
            margin = backend.score(dataset[index])
            if not math.isfinite(margin):
                raise ValueError('Nonfinite NLL margin')
            writer.writerow(dict(sample_id=row.sample_id, participant_id=row.participant_id,
                                 true_label=row.label, nll_margin=margin))
            stream.flush()
            os.fsync(stream.fileno())
            completed.add(row.sample_id)
            if len(completed) % 100 == 0:
                print(path.stem, len(completed), '/', len(manifest), flush=True)
    return len(completed) == len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--model-path')
    parser.add_argument('--dev-only', action='store_true')
    parser.add_argument('--max-minutes', type=float, default=0)
    args = parser.parse_args()
    import torch
    from model import AudioLanguageModel
    from storage import seed_all
    settings = Settings.read(args.settings)
    manifests = {s: load_manifest(args.manifests/f'{s}.csv', s, n)
                 for s, n in [('train', 107), ('dev', 35), ('test', 47)]}
    cohorts = [set(f.participant_id) for f in manifests.values()]
    if any(cohorts[i] & cohorts[j] for i in range(3) for j in range(i)):
        raise ValueError('Participant leakage across Train, Dev and Test')
    import importlib.metadata
    common = dict(settings=asdict(settings),
        manifests={s: sha256(args.manifests/f'{s}.csv') for s in manifests},
        sources={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        versions={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'peft', 'bitsandbytes']},
        evaluation_assets=verify_assets([manifests['dev'], manifests['test']], args.data_root))
    deadline = time.monotonic()+args.max_minutes*60 if args.max_minutes else math.inf
    all_results = []
    with run_lock(args.output):
        seed_all(settings.seed)
        backend = AudioLanguageModel(settings, args.model_path, training=False)
        for checkpoint in args.checkpoints:
            checkpoint_hash = sha256(checkpoint)
            folder = args.output/(checkpoint.stem+'_'+checkpoint_hash[:12])
            folder.mkdir(parents=True, exist_ok=True)
            identity = dict(checkpoint_sha256=checkpoint_hash, **common)
            identity_path = folder/'identity.json'
            if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                raise ValueError('Evaluation source, input or settings changed')
            atomic_json(identity_path, identity)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if payload['identity']['plan']['settings'] != asdict(settings):
                raise ValueError('Evaluation and training model settings differ')
            backend.load(payload['adapter'])
            frozen_path = folder/'frozen_dev_threshold.json'
            for split in ['dev'] if args.dev_only else ['dev', 'test']:
                threshold = None
                if split == 'test':
                    frozen = json.loads(frozen_path.read_text())
                    if frozen['checkpoint_sha256'] != checkpoint_hash or (
                        frozen['dev_scores_sha256'] != sha256(folder/'dev_window_scores.csv')):
                        raise ValueError('Dev threshold identity changed')
                    threshold = frozen['threshold']
                score_path = folder/f'{split}_window_scores.csv'
                if not score_windows(backend, manifests[split], args.data_root, score_path, deadline):
                    print('PAUSED: invoke the same command to resume evaluation', flush=True)
                    return
                people, result = summarize_scores(read_scores(score_path), manifests[split], split, threshold)
                people.to_csv(folder/f'{split}_participant_scores.csv', index=False)
                atomic_json(folder/f'{split}_metrics.json', result)
                if split == 'dev':
                    frozen = dict(threshold=result['threshold'], checkpoint_sha256=checkpoint_hash,
                        dev_scores_sha256=sha256(score_path), criterion='positive F1; min FP; closest 0; higher threshold')
                    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
                        raise ValueError('Frozen Dev threshold changed')
                    atomic_json(frozen_path, frozen)
                all_results.append(dict(checkpoint=checkpoint.name, kind=payload['kind'], **result))
                print(checkpoint.name, result, flush=True)
        pd.DataFrame(all_results).to_csv(args.output/'comparison.csv', index=False)
        for kind in ['independent_local', 'preaggregation_local']:
            rows = [r for r in all_results if r['kind'] == kind and r['split'] == 'test']
            if rows:
                if len(rows) != 3:
                    raise ValueError(f'Supply all three {kind} adapters to calculate their mean')
                atomic_json(args.output/f'{kind}_mean.json', dict(n_clients=3,
                    aggregation='unweighted arithmetic mean of client metrics',
                    **{metric: sum(r[metric] for r in rows)/3
                       for metric in ['f1_positive', 'f1_macro', 'roc_auc', 'average_precision']}))


if __name__ == '__main__':
    main()
