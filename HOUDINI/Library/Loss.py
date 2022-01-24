import torch
from torch import Tensor

# The following two functions borrow from
# https://github.com/havakv/pycox/blob/master/pycox/models/loss.py


def cox_ph_loss_sorted(log_h: Tensor,
                       events: Tensor,
                       reduction: str = 'none',
                       eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    if reduction == 'none':
        return - log_h.sub(log_cumsum_h).mul(events)
    else:
        return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


def cox_ph_loss(log_h: Tensor,
                event_dur: Tensor,
                reduction: str = 'none',
                eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.

    Args:
        log_h: the output of the neural network
        event_dur: the ground-truth label used for training
            including RFSstatus' and 'RFSYears' that correspond to
            events and durations.
        reduction: whether reduce the loss values 
        eps: tolerance 
    """
    events = event_dur[:, 0]
    durations = event_dur[:, 1]
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, reduction, eps)
