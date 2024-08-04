import torch


def get_device():
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
