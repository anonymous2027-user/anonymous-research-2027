"""Sequential, fully participating FL simulation with optimizer-boundary recovery.

The backend and datasets can be replaced by tiny CPU fixtures without loading Qwen.
"""
import json
import signal
import time

import torch

from algorithms import correct_gradients, proximal_penalty, update_control, weighted_average
from schedule import learning_rate, make_units
from storage import capture_rng, cpu_copy, restore_rng, save_checkpoint, seed_all


def run(backend, datasets, method, settings, output, identity, resume=False,
        max_new_steps=0, max_minutes=0, initial=None):
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/'resume.pt'
    if method not in {'central', 'local', 'fedavg', 'fedprox', 'scaffold'}:
        raise ValueError('Unknown method')
    federated = method in {'fedavg', 'fedprox', 'scaffold'}
    if len(datasets) != (1 if method == 'central' else 3):
        raise ValueError('Incorrect number of training datasets')
    sizes = [len(dataset) for dataset in datasets]
    units = make_units(sizes, settings, federated)
    parameters = backend.parameters()
    if resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Resume input, source or configuration identity changed')
    else:
        if checkpoint.exists():
            raise FileExistsError('Output already contains a run; use --resume or a new directory')
        start = cpu_copy(initial if initial is not None else backend.state())
        backend.load(start)
        zeros = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in start.items()}
        state = dict(identity=identity, unit=0, steps=0, windows=0, global_state=start,
            memory=[None]*len(datasets), pending=[], active=None,
            controls=[cpu_copy(zeros) for _ in datasets], server_control=zeros)
        save_checkpoint(output/'initial.pt', dict(adapter=start, identity=identity))
        save_checkpoint(checkpoint, state)
    stop = [False]
    previous_handlers = {}
    for sig in [signal.SIGTERM, signal.SIGINT]:
        previous_handlers[sig] = signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    started, new_steps = time.monotonic(), 0

    def stopping():
        return stop[0] or (max_new_steps > 0 and new_steps >= max_new_steps) or (
            max_minutes > 0 and time.monotonic()-started >= max_minutes*60)

    try:
        while state['unit'] < len(units):
            if stopping():
                return state
            unit = units[state['unit']]
            cid, epoch = unit['client'], unit['epoch']
            optimizer = torch.optim.AdamW([p for _, p in parameters], lr=settings.learning_rate,
                betas=(0.9, 0.999), eps=1e-8, weight_decay=settings.weight_decay)
            active = state['active']
            memory = state['memory'][cid]
            if active is not None:
                backend.load(active['adapter'])
                optimizer.load_state_dict(active['optimizer'])
                restore_rng(active['rng'])
            else:
                start = state['global_state'] if federated or memory is None else memory['adapter']
                backend.load(start)
                if memory is not None and (not federated or memory['epoch'] == epoch):
                    optimizer.load_state_dict(memory['optimizer'])
                    restore_rng(memory['rng'])
                else:
                    seed_all(unit['seed'])
                active = dict(offset=0, learning_rate_sum=0.0, start=cpu_copy(start))
            backend.train(True)
            optimizer.zero_grad(set_to_none=True)
            for offset in range(active['offset'], len(unit['batches'])):
                if stopping():
                    return state
                indices = unit['batches'][offset]
                loss_sum = 0.0
                for index in indices:
                    loss = backend.loss(datasets[cid][index])
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite task loss')
                    (loss/len(indices)).backward()
                    loss_sum += float(loss.detach())
                if method == 'fedprox':
                    proximal_penalty(parameters, active['start'], settings.proximal_mu).backward()
                elif method == 'scaffold':
                    correct_gradients(parameters, state['controls'][cid], state['server_control'])
                lr = learning_rate(epoch*unit['steps_per_epoch']+unit['offset']+offset,
                                   unit['steps_per_epoch'], settings)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], settings.max_grad_norm)
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite gradient')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                active.update(offset=offset+1, learning_rate_sum=active['learning_rate_sum']+lr,
                    adapter=backend.state(), optimizer=cpu_copy(optimizer.state_dict()), rng=capture_rng())
                state['active'] = active
                state['steps'] += 1
                state['windows'] += len(indices)
                new_steps += 1
                save_checkpoint(checkpoint, state)
                print(json.dumps(dict(unit=state['unit']+1, epoch=epoch+1, client=cid,
                    segment=unit['segment']+1, step=state['steps'], windows=state['windows'],
                    loss=loss_sum/len(indices), lr=lr, grad_norm=float(norm))), flush=True)
            # Unit completion is replayable if interrupted between artifact writes and resume save.
            final = cpu_copy(active['adapter'])
            if method == 'scaffold':
                state['controls'][cid] = update_control(active['start'], final, state['controls'][cid],
                    state['server_control'], active['learning_rate_sum'])
            state['memory'][cid] = dict(adapter=final if not federated else None, epoch=epoch,
                optimizer=active['optimizer'], rng=active['rng'])
            if federated:
                state['pending'].append(final)
                round_number = epoch*settings.segments+unit['segment']+1
                if round_number == settings.epochs*settings.segments:
                    save_checkpoint(output/f'local_before_final_client_{cid}.pt',
                        dict(adapter=final, identity=identity, kind='preaggregation_local', client=cid))
                if cid == len(datasets)-1:
                    state['global_state'] = weighted_average(state['pending'], sizes)
                    state['pending'] = []
                    if method == 'scaffold':
                        state['server_control'] = weighted_average(state['controls'], sizes)
                    save_checkpoint(output/f'global_round_{round_number:02d}.pt',
                        dict(adapter=state['global_state'], identity=identity, kind='global', round=round_number))
            elif epoch == settings.epochs-1:
                name = 'central_final.pt' if method == 'central' else f'local_only_client_{cid}.pt'
                save_checkpoint(output/name, dict(adapter=final, identity=identity,
                    kind='central' if method == 'central' else 'independent_local', client=cid))
            state['unit'] += 1
            state['active'] = None
            save_checkpoint(checkpoint, state)
        expected = settings.epochs*sum(sizes)
        if state['windows'] != expected:
            raise ValueError(f'Wrong total window presentations: {state["windows"]} != {expected}')
        return state
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
