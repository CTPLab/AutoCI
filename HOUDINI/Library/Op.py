import torch


class NotHandledException(Exception):
    pass


def _list_to_tensor(input: torch.autograd.Variable,
                    chunk: int = 1,
                    spl_dim: int = 1,
                    cat_dim: int = 0) -> torch.autograd.Variable:
    """Convert a pseudo list (tensor) with Batch x Env x Feat to a 
    tensor with (Batch * Env) x Feat, such conversion is more efficient for 
    the follow-up computation with the tensor as input.

    Args:
        input: pytorch tensor with Batch x Env x Feat
        chunk: the size along the splitted dim for each element in the
            intermediate list, mostly = 1
        spl_dim: the dim along which the split operation performs
        cat_dim: the dim along which the cat operation performs

    Returns:
        pytorch tensor with (Batch * Env) x Feat

    """

    input_list = torch.split(input, chunk, dim=spl_dim)
    output = torch.cat(input_list, dim=cat_dim).squeeze(dim=1)
    return output


def _tensor_to_list(input: torch.autograd.Variable,
                    batch: int,
                    spl_dim: int = 0,
                    stk_dim: int = 1) -> torch.autograd.Variable:
    """Convert a tensor with (Batch * Env) x Feat to 
    a pseudo list (tensor) with Batch x Env x Feat.

    Args:
        input: pytorch tensor with (Batch * Env) x Feat
        batch: the batch size along the splitted dim for each element in the
            intermediate list
        spl_dim: the dim along which the split operation performs
        stk_dim: the dim along which the stack operation performs

    Returns:
        pytorch tensor with Batch x Env x Feat

    """

    input_list = torch.split(input, batch, dim=spl_dim)
    output = torch.stack(input_list, dim=stk_dim)
    return output


def pp_repeat(num, fn):
    def ret(x):
        for _ in range(num):
            x = fn(x)
        return x
    return ret


def pp_compose(g, f):
    return lambda x: g(f(x))


def pp_cat(fn):
    def ret(x):
        interm, iarg = fn(x)
        out = _list_to_tensor(interm)
        return out, iarg
    return ret


def pp_do(fn):
    # only allow do function once
    def ret(iterable):
        if type(iterable) == tuple:
            iterable, iarg = iterable
        else:
            iarg = None

        assert isinstance(iterable, torch.autograd.Variable), \
            'the input type {} is not torch variable'.format(type(iterable))

        batch = iterable.shape[0]
        iterable = _list_to_tensor(iterable)
        interm, iarg = fn((iterable, iarg))
        output = _tensor_to_list(interm, batch)
        return output, iarg
    return ret


def pp_map(fn):
    def ret(iterable):
        if type(iterable) == tuple:
            iterable, iarg = iterable
        else:
            iarg = None

        assert isinstance(iterable, torch.autograd.Variable), \
            'the input type {} is not torch variable'.format(type(iterable))

        batch = iterable.shape[0]
        iterable = _list_to_tensor(iterable)
        interm, iarg = fn((iterable, iarg))
        output = _tensor_to_list(interm, batch)
        return output, iarg
    return ret


def pp_conv(fn):
    def ret(iterable):
        if type(iterable) == tuple:
            iterable, iarg = iterable
        else:
            iarg = None

        assert isinstance(iterable, torch.autograd.Variable), \
            'the input type {} is not torch variable'.format(type(iterable))

        batch = iterable.shape[0]
        iterable = _list_to_tensor(iterable)
        interm, iarg = fn((iterable, iarg))
        output = _tensor_to_list(interm, batch)
        return output, iarg
    return ret


def pp_reduce(fn):
    def ret(iterable):
        if type(iterable) == tuple:
            iterable, iarg = iterable
        else:
            iarg = None

        assert isinstance(iterable, torch.autograd.Variable), \
            'the input type {} is not torch variable'.format(type(iterable))

        batch = iterable.shape[0]
        iterable = _list_to_tensor(iterable)
        interm, iarg = fn((iterable, iarg))
        output = _tensor_to_list(interm, batch)
        return output, iarg
    return ret
