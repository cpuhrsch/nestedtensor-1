import torch


def embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2,
                  scale_grad_by_freq=False, mode='mean', sparse=False,
                  per_sample_weights=None, include_last_offset=False):
    # Check for backward compatibility.
    # Used to be embedding_bag(weight, input, ...)
    # Now is     embedding_bag(input, weight, ...)
    if weight.dtype == torch.long and input.is_floating_point():
        warnings.warn("Argument order of nn.functional.embedding_bag was changed. "
                      "Usage `embedding_bag(weight, input, ...)` is deprecated, "
                      "and should now be `embedding_bag(input, weight, ...)`.")
        weight, input = input, weight

    if per_sample_weights is not None and input.nested_size() != per_sample_weights.nested_size():
        raise ValueError("embedding_bag: If per_sample_weights ({}) is not None, "
                         "then it must have the same nested_size as the input ({})"
                         .format(per_sample_weights.nested_size(), input.nested_size()))

    if max_norm is not None:
        raise NotImplementedError(
            "max_norm kwarg is currently not supported, please file an issue on https://github.com/pytorch/nestedtensor")

    if input.dim() != 2:
        raise ValueError(
            "Input is expected to be of dimension 2, got {} instead.".format(input.dim()))

    if offsets is not None:
        raise ValueError("offsets must be None.")
    offsets = torch.tensor([0], dtype=torch.int64, device=input.device)

    if mode == 'sum':
        mode_enum = 0
    elif mode == 'mean':
        mode_enum = 1
    elif mode == 'max':
        mode_enum = 2

        if scale_grad_by_freq:
            raise ValueError(
                "max mode does not support scaling the gradient by the frequency")

        if sparse:
            raise ValueError("max mode does not support sparse weights")

    else:
        raise ValueError("mode has to be one of sum, mean or max")

    ret, _, _, _ = torch.embedding_bag(
        weight,
        input,
        offsets,
        scale_grad_by_freq,
        mode_enum,
        sparse,
        per_sample_weights,
        include_last_offset)
    return ret
