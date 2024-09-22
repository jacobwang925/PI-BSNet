import torch
import torch.nn as nn

# Define the neural network architecture
class MyConvNet(nn.Module):
    def __init__(self, dims = (2,8,8),pad =1, kernel =3):
        super(MyConvNet,self).__init__()
        self.size = int(torch.prod(torch.tensor(dims)))
        self.dims = dims
        self.pad = pad
        self.kernel = kernel
        
        self.linear = nn.Linear(1, self.size)
        self.nonlinear = nn.ReLU(inplace=True)
        self.Conv2d = nn.Conv2d(self.dims[0],1, kernel, stride=1, padding=pad, padding_mode="replicate", dilation=1, bias=True)
    
    def output_shape(self):
        dilation =1
        stride =1
        def out(h_in):
           a = (h_in+self.pad*2 - dilation*(self.kernel-1) -1 )/stride +1
           return int(a)
        
        return (out(self.dims[1]), out(self.dims[2]))
       
    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        x = self.Conv2d(x.view(-1,*self.dims))
        x = self.nonlinear(x)
        x = torch.nn.Flatten()(x)
        return x
    
class ControlPointNet(nn.Module):
    def __init__(self, n_cp_x, n_cp_t, hidden_dim=64, hidden_depth=1, conv2d =False):
        super(ControlPointNet, self).__init__()
        
        layers = [nn.Linear(1, hidden_dim), nn.ReLU(inplace=True)]
        out_shape =hidden_dim
        
        self.use_conv2d = conv2d
        if conv2d:
            conv2d = MyConvNet()
            out_shape = torch.prod(torch.tensor(conv2d.output_shape()))
            layers =[conv2d]
        
        for i in range(hidden_depth):
            layers.append(nn.Linear(out_shape, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Linear(hidden_dim, (n_cp_x - 1) * (n_cp_t - 1)))
        
        self.nn  = torch.nn.ModuleList(layers)
        self.n_cp_x = n_cp_x
        self.n_cp_t = n_cp_t

    def forward(self, lambda_param):
        x = lambda_param
        for enum, layer in enumerate(self.nn):
            x = layer(x)
            
        return x.view(-1, self.n_cp_t - 1, self.n_cp_x - 1)
