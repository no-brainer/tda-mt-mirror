from itertools import repeat


def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader
