import torch
from torch.nn import Sequential, Linear, ReLU
from torch.optim import Adam
import numpy as np
from utils import BsKnots, BsKnots_derivatives
from pathlib import Path
import random

N = 3 # number of centroid locations per dimension
M =11 # number of output control points per dimension
D =0.09 # diffusion constant
time_steps = 9

#calculate B-splines
B=41
d=3

tk_t, Ln_t, Bit_t = BsKnots(time_steps, d, B)
tk_x, Ln_x, Bit_x = BsKnots(M, d, B)
Bit_t_d, Bit_t_dd = BsKnots_derivatives(time_steps, d, B, Ln_t, tk_t)
Bit_x_d, Bit_x_dd = BsKnots_derivatives(M,d, B,Ln_x, tk_x)


#define a model
model = Sequential(*[Linear(4,92),
                     ReLU(inplace=True), 
                     Linear(92, 92),
                     ReLU(inplace=True),  
                     Linear(92, (M**3)*time_steps )])


#define optimizer
opt = Adam(model.parameters(), lr = 0.0001)



# compute gaussian
def make_gaussian(cw, x= torch.linspace(0, 1, 12)):
        grid_x, grid_y, grid_z = torch.meshgrid(x,x,x)
        cx, cy, cz, w = cw[:, 0], cw[:, 1], cw[:, 2],cw[:, 3] 
        
        # Reshape grid tensors to match the dimensions for broadcasting
        cx = cx.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        cy = cy.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        cz = cz.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w = w.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        grid_x = grid_x.unsqueeze(0).repeat(cw.size(0), 1, 1, 1) 
        grid_y = grid_y.unsqueeze(0).repeat(cw.size(0), 1, 1, 1)  
        grid_z = grid_z.unsqueeze(0).repeat(cw.size(0), 1, 1, 1) 
        
        r2 =((grid_x - cx) ** 2 +(grid_y - cy) ** 2 + (grid_z - cz) ** 2)
        
        return torch.exp(-r2 / (2 * w ** 2))/torch.sqrt(2*torch.pi*w**2)
    

# load input data
x = torch.linspace(.4,.5,N)
width = torch.linspace(0.1,0.2,N**3)
x0, y0, z0 = torch.meshgrid(x,x,x)
x_ = torch.vstack([x0.reshape(-1), y0.reshape(-1), z0.reshape(-1), width]).T


ctrl_loc= torch.linspace(0, 1, M)


#make truth data at every
time = torch.linspace(0,1,time_steps)
true_y = []
for i in range(time_steps):
    #replace with new width.
    w_i = torch.sqrt(width**2 +2*D*time[i])
    x_new = x_
    x_new[:,3] = w_i
    true_y.append(make_gaussian(x_new, ctrl_loc))
true_y = torch.stack(true_y, dim =0)
true_y = true_y.reshape(-1,time_steps*M**3)

def pde_observer(pred_y, true_y):
        
    pred_y_= pred_y.reshape(N**3,time_steps,M,M,M)
    true_y_= true_y.reshape(N**3,time_steps,M,M,M)
    #get first time derivative
    pred_y_t_d = torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz", pred_y_, Bit_t_d, Bit_x, Bit_x, Bit_x)
    pred_y_x_dd = torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz", pred_y_, Bit_t, Bit_x_dd, Bit_x, Bit_x)
    pred_y_x_dd += torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz", pred_y_, Bit_t, Bit_x, Bit_x_dd, Bit_x)
    pred_y_x_dd += torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz", pred_y_, Bit_t, Bit_x, Bit_x, Bit_x_dd)
    
    true_y_t_d = torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz", true_y_, Bit_t_d, Bit_x, Bit_x, Bit_x)
    true_y_x_dd = torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz", true_y_, Bit_t, Bit_x_dd, Bit_x, Bit_x)
    true_y_x_dd += torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz", true_y_, Bit_t, Bit_x, Bit_x_dd, Bit_x)
    true_y_x_dd += torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz", true_y_, Bit_t, Bit_x, Bit_x, Bit_x_dd)
    
    return (torch.mean((pred_y_t_d - true_y_t_d)**2),
            torch.mean((pred_y_x_dd - true_y_x_dd)**2),
            torch.mean((pred_y_t_d - pred_y_x_dd)**2),
            torch.mean((true_y_t_d - true_y_x_dd)**2))

#TRAIN
G=1021
print(f"\n\n train for {G}\n\n")
for i in range(G):
    opt.zero_grad()
    pred_y = model(x_)
    
    diff_t_d, diff_x_dd, heat_pred, heat_true =  pde_observer(pred_y, true_y)
    
    loss=heat_pred
    if i%10==0:
        j_select=random.sample(range(pred_y.shape[0]),2)
        k_select=random.sample(range(pred_y.shape[1]),M**3//2)
        data_loss = torch.mean((pred_y[j_select][ :,k_select] - true_y[j_select][:,k_select])**2)
        loss += data_loss
    
    loss.backward()
    opt.step()
    if i%5==0:
        print(f"\t\t{i:09d} Loss is now: {loss.item():2.8f}:  {data_loss} {diff_t_d} {diff_x_dd} {heat_pred} {heat_true}", end="\r")
        
print()


# Save new checkpoint
checkpoint_filename = f"scratch_loss_{loss:.8f}.pt"
checkpoint_path = Path("models/3dfp/") / checkpoint_filename
torch.save({"model_state_dict":model.state_dict()
            }, checkpoint_path)
#EVALUATE

pred_y_= pred_y.detach().reshape(N**3,time_steps,M,M,M)
true_y_= true_y.detach().reshape(N**3,time_steps,M,M,M)

pred_y_spline = torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz", pred_y_, Bit_t, Bit_x, Bit_x, Bit_x)
true_y_spline = torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz", true_y_, Bit_t, Bit_x, Bit_x, Bit_x)

import matplotlib.pyplot as plt
i=5
fig, ax = plt.subplots(1,2, figsize=(15,5))

ax[0].matshow(pred_y_[i,0,:,:,M//2])
ax[1].matshow(true_y_[i,0,:,:,M//2])


fig, ax = plt.subplots(1,2, figsize=(15,5))

ax[0].matshow(pred_y_spline[i,0,:,:,B//2])
ax[1].matshow(true_y_spline[i,0,:,:,B//2])

fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].matshow(pred_y_[i,:,:,M//2,M//2])
ax[1].matshow(true_y_[i,:,:,M//2,M//2])


fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].matshow(pred_y_spline[i,:,:,B//2,B//2])
ax[1].matshow(true_y_spline[i,:,:,B//2,B//2])



plt.show()

