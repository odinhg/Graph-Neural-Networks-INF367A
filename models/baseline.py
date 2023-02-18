import torch.nn as nn
from .utils import init_weights

class BaseLineModel(nn.Module):
    def __init__(self, input_nodes=98, dropout_p=0.2, batchnorm_eps=0.1):
        super().__init__()
        self.fcnn = nn.Sequential(
                        nn.Linear(input_nodes, input_nodes*4),
                        nn.BatchNorm1d(input_nodes*4, batchnorm_eps),
                        nn.ReLU(),
                        nn.Dropout(dropout_p),
                        nn.Linear(input_nodes*4, input_nodes*4),
                        nn.BatchNorm1d(input_nodes*4, batchnorm_eps),
                        nn.ReLU(),
                        nn.Linear(input_nodes*4, input_nodes-3),
                    )
        self.fcnn.apply(init_weights)

    def forward(self, x):
        x = self.fcnn(x)
        return x
