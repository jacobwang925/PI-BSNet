import torch
import torch.nn as nn

#activation registry
activations ={"relu": torch.relu, 
              "sigmoid":torch.sigmoid, 
              "tanh":torch.tanh }

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


# Define the neural network architecture
class ControlPointNetA(nn.Module):
    def __init__(self, 
                 input_size:int, 
                 n_ctrl_pts_time:int, 
                 n_ctrl_pts_state:int,
                 dimension:int = 3,
                 hidden_dim:int =128, 
                 hidden_depth:int =5, 
                 activation="relu"):
        
        super(ControlPointNetA, self).__init__()
        self.name="RecurrentModel"
        self.hidden_dim = hidden_dim
        self.hidden_depth=hidden_depth
        self.n_ctrl_pts_time = n_ctrl_pts_time
        self.n_ctrl_pts_state = n_ctrl_pts_state
        self.act = activations.get(activation)
        if self.act == None: 
            raise NotImplementedError(f"activation {activation} does not exist")
        self.output_size= (n_ctrl_pts_state**dimension) 
        self.output_shape = [n_ctrl_pts_state for i in range(dimension)]
        
        
        
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.h2x = nn.Linear(hidden_dim, hidden_dim)
        # Predict control points for the entire grid except for the initial time step
        self.decode = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.encode = nn.Linear(2 * hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, self.output_size)
        
        self.apply(init_weights)

    def forward(self, par):
        par = self.act(self.fc1(par))
        par = self.act(self.fc2(par))
        x = self.act(self.h2x(par))
        output_list = []
        for i in range(self.n_ctrl_pts_time):
            dec = self.act(self.decode(x))
            enc = self.act(self.encode(dec))
            x = enc + x
            surf = self.out(x)
            output_list.append(surf)
            x = self.act(self.h2x(x))

        output = torch.stack(output_list, dim=1)
        return (
            output.view(-1, self.n_ctrl_pts_time, *self.output_shape),
            output_list[0].reshape(-1, *self.output_shape)
            )

class ControlPointNetB(nn.Module):
    def __init__(self, 
                 input_size:int, 
                 n_ctrl_pts_time:int, 
                 n_ctrl_pts_state:int,
                 dimension:int,  
                 hidden_dim:int =128, 
                 hidden_depth:int =5, 
                 activation="relu"):
        
        super(ControlPointNetB, self).__init__()
        self.name="SimpleModel"
        self.hidden_dim = hidden_dim
        self.hidden_depth=hidden_depth
        self.n_ctrl_pts_time = n_ctrl_pts_time
        self.n_ctrl_pts_state = n_ctrl_pts_state
        self.act = activations.get(activation)
        if self.act == None: 
            raise NotImplementedError(f"activation {activation} does not exist")
       
        self.output_size= n_ctrl_pts_time* (n_ctrl_pts_state**dimension) 
        self.output_shape = [n_ctrl_pts_time]+[n_ctrl_pts_state for i in range(dimension)]
        
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.hidden_list = [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_dim)]
        self.out = nn.Linear(hidden_dim, self.output_size)
        self.apply(init_weights)

    def forward(self, par):
        x = self.act(self.fc1(par))
        for i in range(self.hidden_depth):
            x = self.act(self.hidden_list[i](x))
        output = self.out(x).reshape(-1,*self.output_shape)
        return (output, output[:,0,:])
