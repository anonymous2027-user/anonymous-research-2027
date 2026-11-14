"""Fixed coverage with segment-level communication boundaries, effective batch 21."""
import math
import numpy as np


def epoch_plan(size, epoch, client, settings, federated=True):
    seed = settings.seed + (epoch+1)*100003 + client*1009 if federated else settings.seed+client*1009
    order = np.arange(size)
    np.random.default_rng(seed+epoch*1009).shuffle(order)
    if federated:
        old = [order[i:i+64].tolist() for i in range(0, size, 64)]
        chunks = np.array_split(np.arange(len(old)), settings.segments)
        groups = [[index for batch in chunk for index in old[int(batch)]] for chunk in chunks]
    else:
        groups = [order.tolist()]
    result, offset = [], 0
    for group in groups:
        batches = [group[i:i+settings.batch] for i in range(0, len(group), settings.batch)]
        if not batches:
            raise ValueError('Empty training segment')
        result.append(dict(seed=seed, batches=batches, offset=offset))
        offset += len(batches)
    return result


def learning_rate(step, steps_per_epoch, settings):
    warmup = int(round(steps_per_epoch*settings.warmup_epochs))
    if warmup and step < warmup:
        factor = (step+1)/warmup
    else:
        progress = (step-warmup)/max(settings.epochs*steps_per_epoch-warmup, 1)
        factor = (1+math.cos(math.pi*min(max(progress, 0.0), 1.0)))/2
    return settings.learning_rate*factor


def make_units(sizes, settings, federated=True):
    units = []
    for epoch in range(settings.epochs):
        plans = [epoch_plan(size, epoch, cid, settings, federated) for cid, size in enumerate(sizes)]
        for segment in range(settings.segments if federated else 1):
            for cid, plan in enumerate(plans):
                unit = dict(epoch=epoch, segment=segment, client=cid, **plan[segment])
                unit['steps_per_epoch'] = sum(len(g['batches']) for g in plan)
                units.append(unit)
    return units
