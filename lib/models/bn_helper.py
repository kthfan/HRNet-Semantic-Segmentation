import torch
import functools


if torch.cuda.device_count() > 1:
    if torch.__version__.startswith('0'):
        from .sync_bn.inplace_abn.bn import InPlaceABNSync
        BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
        BatchNorm2d_class = InPlaceABNSync
        relu_inplace = False
    else:
        BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
        relu_inplace = True
else:
    BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d
    relu_inplace = True
