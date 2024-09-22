import torch
import torch.nn as nn

# Define the neural network architecture
class ControlPointNet(nn.Module):
    def __init__(self, n_cp_x, n_cp_t, hidden_dim=64, hidden_depth=1):
        super(ControlPointNet, self).__init__()
        layers = [nn.Linear(1, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Linear(hidden_dim, (n_cp_x - 1) * (n_cp_t - 1)))
        self.nn  = torch.nn.Sequential(*layers)
        self.n_cp_x = n_cp_x
        self.n_cp_t = n_cp_t

    def forward(self, lambda_param):
        x = self.nn(lambda_param)
        return x.view(self.n_cp_t - 1, self.n_cp_x - 1)
