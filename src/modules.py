import torch
import torch.nn as nn


class ControlPointNet(nn.Module):
    def __init__(self, n_cp_x, n_cp_t, hidden_dim=64, hidden_depth=1):
        super(ControlPointNet, self).__init__()

        layers = [nn.Linear(1, hidden_dim), nn.ReLU(inplace=True)]
        out_shape = hidden_dim

        
        for i in range(hidden_depth):
            layers.append(nn.Linear(out_shape, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, (n_cp_x - 1) * (n_cp_t - 1)))

        self.nn = torch.nn.ModuleList(layers)
        self.n_cp_x = n_cp_x
        self.n_cp_t = n_cp_t

    def forward(self, lambda_param):
        x = lambda_param
        for enum, layer in enumerate(self.nn):
            x = layer(x)

        return x.view(-1, self.n_cp_t - 1, self.n_cp_x - 1)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


# Define the neural network architecture
class ControlPointNet3D(nn.Module):
    def __init__(self, n_cp_t, n_cp_x, n_cp_y, n_cp_z, hidden_dim=128):
        super(ControlPointNet3D, self).__init__()
        self.fc1 = nn.Linear(4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.h2x = nn.Linear(hidden_dim, hidden_dim)
        # Predict control points for the entire grid except for the initial time step
        self.decode = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.encode = nn.Linear(2 * hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, n_cp_x * n_cp_y * n_cp_z)
        self.n_cp_t = n_cp_t
        self.n_cp_x = n_cp_x
        self.n_cp_y = n_cp_y
        self.n_cp_z = n_cp_z
        self.apply(init_weights)

    def forward(self, par):
        par = torch.relu(self.fc2(torch.relu(self.fc1(par))))
        x = torch.relu(self.h2x(par))
        output_list = []
        for i in range(self.n_cp_t):
            dec = torch.relu(self.decode(x))
            enc = torch.relu(self.encode(dec))
            x = enc + x
            surf = self.out(x)
            output_list.append(surf)
            x = torch.relu(self.h2x(x))

        output = torch.stack(output_list, dim=1)
        return (
            output.view(-1, self.n_cp_t, self.n_cp_x, self.n_cp_y, self.n_cp_z),
            output_list[0].reshape(-1, self.n_cp_x, self.n_cp_y, self.n_cp_z),
        )
