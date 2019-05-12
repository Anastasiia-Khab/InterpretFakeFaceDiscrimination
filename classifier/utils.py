import numpy as np
import os

import torch

def save_model(model, optimizer, epoch, iter, chkp_dir):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'iter': iter
    }
    torch.save(state, os.path.join(chkp_dir, 'epoch-{}.chkp'.format(epoch)))


def load_model(chkp_dir, chkp_name):
    state = torch.load(os.path.join(chkp_dir, chkp_name))
    return state
