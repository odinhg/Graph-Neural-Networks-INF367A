import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
