"""FedAvg, adapter FedProx, and an explicitly documented AdamW SCAFFOLD variant."""
import math

import torch


def weighted_average(states, sizes):
    if len(states) != len(sizes) or not states or any(size <= 0 for size in sizes):
        raise ValueError('Expected matching nonempty states and positive client sizes')
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise ValueError('Adapter/control keys differ')
    total = sum(sizes)
    result = {}
    for key in sorted(keys):
        reference = states[0][key]
        accumulator = torch.zeros_like(reference, device='cpu', dtype=torch.float32)
        for state, size in zip(states, sizes):
            value = state[key]
            if value.shape != reference.shape or not torch.isfinite(value).all():
                raise ValueError('Invalid adapter shape or nonfinite values')
            accumulator.add_(value.detach().cpu().float(), alpha=size/total)
        result[key] = accumulator.to(reference.dtype)
    return result


def proximal_penalty(named_parameters, global_state, mu):
    if mu < 0:
        raise ValueError('Negative proximal coefficient')
    values = [((parameter.float() - global_state[key].to(parameter.device).float())**2).sum()
              for key, parameter in named_parameters]
    if not values:
        raise ValueError('No trainable adapter parameters')
    return (mu / 2) * torch.stack(values).sum()


def correct_gradients(named_parameters, client_control, server_control):
    # Exactly once per accumulated optimizer update, before clipping and AdamW.
    with torch.no_grad():
        for name, parameter in named_parameters:
            if parameter.grad is None:
                raise ValueError(f'Adapter has no task gradient: {name}')
            correction = server_control[name] - client_control[name]
            parameter.grad.add_(correction.to(device=parameter.device, dtype=parameter.grad.dtype))


def update_control(start, end, client_control, server_control, learning_rate_sum):
    if not math.isfinite(learning_rate_sum) or learning_rate_sum <= 0:
        raise ValueError('Control update needs a positive actual learning-rate sum')
    # Option-II-style displacement update. With constant LR, sum(LR) = K * LR.
    result = {key: client_control[key].float() - server_control[key].float()
              + (start[key].float() - end[key].float()) / learning_rate_sum for key in start}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError('Nonfinite control state')
    return result
