"""Train centralized, independent local-only, FedAvg, FedProx or SCAFFOLD adapters."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import importlib.metadata
from pathlib import Path

from config import Settings
from data import AudioTextDataset, load_manifest, verify_assets
from partitions import PARTITIONS
from schedule import make_units
from storage import atomic_json, sha256


@contextmanager
def run_lock(output):
    # CUDA training is supported on Linux. A held flock is released even after process death.
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output/'run.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=['central', 'local', 'fedavg', 'fedprox', 'scaffold'])
    parser.add_argument('--partition', choices=list(PARTITIONS), default='iid')
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--partitions', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--model-path', help='An immutable local snapshot of the pinned model revision')
    parser.add_argument('--initial', type=Path, help='Optional trusted common initialization checkpoint')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--max-new-steps', type=int, default=0)
    parser.add_argument('--max-minutes', type=float, default=0)
    args = parser.parse_args()
    settings = Settings.read(args.settings)
    source = load_manifest(args.manifests/'train.csv', 'train', 107)
    dev = load_manifest(args.manifests/'dev.csv', 'dev', 35)
    test = load_manifest(args.manifests/'test.csv', 'test', 47)
    cohorts = [set(frame.participant_id) for frame in [source, dev, test]]
    if any(cohorts[i] & cohorts[j] for i in range(3) for j in range(i)):
        raise ValueError('Participant leakage between Train, Dev and Test')
    if [int(f.groupby('participant_id').label.first().sum()) for f in [source, dev, test]] != [30, 12, 14]:
        raise ValueError('Incorrect official split class counts')
    paths = [args.manifests/f'{split}.csv' for split in ['train', 'dev', 'test']]
    if args.method == 'central':
        frames = [source]
    else:
        if args.partitions is None:
            parser.error('--partitions is required for local or federated training')
        client_paths = [args.partitions/args.partition/f'client_{cid}.csv' for cid in range(3)]
        frames = [load_manifest(path, 'train', counts[0])
                  for path, counts in zip(client_paths, PARTITIONS[args.partition])]
        source_rows = source.set_index('sample_id')
        seen = set()
        for frame, (total, positive) in zip(frames, PARTITIONS[args.partition]):
            people = set(frame.participant_id)
            if seen & people or frame.groupby('participant_id').label.first().sum() != positive:
                raise ValueError('Wrong partition counts or overlapping participants')
            seen.update(people)
            import pandas as pd
            pd.testing.assert_frame_equal(frame.reset_index(drop=True),
                source_rows.loc[frame.sample_id].reset_index()[frame.columns], check_dtype=False)
        if seen != set(source.participant_id):
            raise ValueError('Clients do not exhaust Train')
        paths.extend(client_paths)
    federated = args.method in {'fedavg', 'fedprox', 'scaffold'}
    units = make_units([len(f) for f in frames], settings, federated)
    plan = dict(method=args.method, partition=None if args.method == 'central' else args.partition,
        settings=asdict(settings), training_windows=settings.epochs*sum(map(len, frames)),
        optimizer_steps=sum(len(u['batches']) for u in units),
        aggregations=settings.epochs*settings.segments if federated else 0)
    print(plan, flush=True)
    if args.plan_only:
        return
    import torch
    from engine import run
    from model import AudioLanguageModel
    from storage import seed_all
    identity = dict(plan=plan, manifests=[sha256(p) for p in paths],
        source={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        training_assets=verify_assets(frames, args.data_root),
        versions={n: importlib.metadata.version(n) for n in
                  ['torch', 'transformers', 'peft', 'bitsandbytes', 'numpy', 'pandas']},
        initial_sha256=sha256(args.initial) if args.initial else None)
    # Full asset hashes are intentionally calculated before loading the expensive GPU model.
    with run_lock(args.output):
        existing = args.output/'experiment.json'
        if existing.exists():
            import json
            if json.loads(existing.read_text()) != identity or not args.resume:
                raise ValueError('Output directory belongs to another run; choose a new output')
        else:
            if args.resume:
                raise FileNotFoundError('Cannot resume without experiment.json')
            atomic_json(existing, identity)
        seed_all(settings.seed)
        backend = AudioLanguageModel(settings, args.model_path, training=True)
        initial = None
        if args.initial:
            initial_checkpoint = torch.load(args.initial, map_location='cpu', weights_only=False)
            initial = initial_checkpoint.get('adapter', initial_checkpoint.get('trainable_state_dict'))
            if initial is None:
                raise ValueError('Initialization has no adapter tensors')
        result = run(backend, [AudioTextDataset(f, args.data_root) for f in frames], args.method,
            settings, args.output, identity, args.resume, args.max_new_steps, args.max_minutes, initial)
        print('COMPLETE' if result['unit'] == len(units) else 'PAUSED at optimizer boundary', flush=True)


if __name__ == '__main__':
    main()
