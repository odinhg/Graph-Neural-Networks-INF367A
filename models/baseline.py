import torch.nn as nn
from .utils import init_weights

class BaseLineModel(nn.Module):
    def __init__(self, input_nodes=98):
        super().__init__()
        self.fcnn = nn.Sequential(
                        nn.Linear(input_nodes, 8*input_nodes),
                        nn.BatchNorm1d(8*input_nodes),
                        nn.ReLU(),

                        nn.Linear(8*input_nodes, 4*input_nodes),
                        nn.BatchNorm1d(4*input_nodes),
                        nn.ReLU(),

                        nn.Linear(4*input_nodes, 2*input_nodes),
                        nn.BatchNorm1d(2*input_nodes),
                        nn.ReLU(),

                        nn.Linear(2*input_nodes, input_nodes),
                        nn.BatchNorm1d(input_nodes),
                        nn.ReLU(),

                        nn.Linear(input_nodes, input_nodes-3),
                    )
        self.fcnn.apply(init_weights)

    def forward(self, x):
        x = self.fcnn(x)
        return x
