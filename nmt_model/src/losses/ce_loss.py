import torch.nn.functional as F


class CELoss:

    def __init__(self):
        pass

    def __call__(self, outputs, trg_enc, *args, **kwargs):
        return F.cross_entropy(
            outputs.transpose(1, 2)[:, :, :-1],
            trg_enc[:, 1:]
        )
