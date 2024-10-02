import torch
import numpy as np
import matplotlib.pyplot as plt
from modules import ControlPointNet3D
from base_run import base_run
from scripts.bs_net_fokker_planck2 import bsnet_train

#a = np.load("data/gaussian_random_30_0.25_0.75_0.03_0.20.npy")
#a = torch.tensor(a.astype(np.float32))
#filename = "models/3dfp/v0/20240930_095023_loss_0.000845.pt"



#filename = "models/3dfp/v0/20240930_202630_loss_0.07793247.pt"
filename = "models/3dfp/v1/20241001_164929_loss_9.44688225.pt"
data = torch.load(filename)

bs = bsnet_train()
#bs.setup("scripts/config_fp_old.json")
bs.setup("scripts/config_fp.json")#, n_points = 51)


Ns = bs.n_points
ncpx = bs.n_ctrl_pts_state
ncpt = bs.n_ctrl_pts_time

bs.model.load_state_dict(data["model_state_dict"])
bs.model.eval()


true_init_condition_train = bs.initial_conditions_from_lambda(bs.lambda_train)
true_init_condition_train[:,:,:,[0,-1]]=0
true_init_condition_train = bs.make_surface(true_init_condition_train)

true_init_condition_test = bs.initial_conditions_from_lambda(bs.lambda_test)
true_init_condition_test [:,:,:,[0,-1]]=0
true_init_condition_test = bs.make_surface(true_init_condition_test)

#run model
y_hat_train, y_ic_train = bs.model(bs.lambda_train)
y_hat_train[:,:,:,:,[0,-1]]=1
y_ic_train[:,:,:,[0,-1]] =1

pred_init_condition_train = bs.make_surface(y_ic_train)
print(y_hat_train.shape, y_ic_train.unsqueeze(1).shape)
#y_hat_train = torch.cat([y_ic_train.unsqueeze(1),y_hat_train], dim=1)


y_hat_test, y_ic_test = bs.model(bs.lambda_test)

y_hat_test[:,:,:,:,[0,-1]]=1
y_ic_test[:,:,:,[0,-1]] =1
print("---->>>", torch.sum(y_hat_test[:,:,:,:,[0,-1]]))
pred_init_condition_test = bs.make_surface(y_ic_test)
#y_hat_test = torch.cat([y_ic_test.unsqueeze(1),y_hat_test], dim=1)


pick = 5
plt.gca().xaxis.tick_bottom()

#1
fig, ax = plt.subplots(2,2, figsize=(15,15))

diff_ic_train = ((true_init_condition_train-pred_init_condition_train)/(true_init_condition_train+0.000001)).detach().numpy()
mean_diff_ic_train = np.mean(diff_ic_train**2, axis=(1,2,3))
std_diff_ic_train = np.std(diff_ic_train**2, axis=(1,2,3)) 
ax[0,0].scatter(np.linspace(1,len(mean_diff_ic_train),len(mean_diff_ic_train)), mean_diff_ic_train, label = "Train")
#x[0,0].errorbar(np.linspace(1,len(mean_diff_ic_train),len(mean_diff_ic_train)), mean_diff_ic_train, std_diff_ic_train)

diff_ic_test = (true_init_condition_test-pred_init_condition_test/(true_init_condition_test+0.000001)).detach().numpy()
mean_diff_ic_test = np.mean(diff_ic_test**2, axis=(1,2,3))
std_diff_ic_test = np.std(diff_ic_test**2, axis=(1,2,3)) 
ax[0,0].scatter(np.linspace(1,len(mean_diff_ic_test),len(mean_diff_ic_test)), mean_diff_ic_test, label = "Test")
#ax[0,0].errorbar(np.linspace(1,len(mean_diff_ic_test),len(mean_diff_ic_test)), mean_diff_ic_test, std_diff_ic_test)

ax[0,0].set_title("<$(I.C._{pred} - I.C._{true})^2$>")

ax01 = ax[0,1].matshow((true_init_condition_test-pred_init_condition_test).detach()[pick,:,:, Ns//2])
plt.colorbar(ax01)
ax[0,1].set_title(" $1^{st}$ test point $(I.C._{pred} - I.C._{true})^2$ (XY plane)")

ax02 = ax[1,0].matshow((pred_init_condition_test).detach()[pick,:,:, Ns//2].T)
plt.colorbar(ax02)
ax[1,0].set_title(" $1^{st}$ test point $I.C._{pred}$ (XY plane)")
ax[1,0].set_xlabel("X (cm)")
ax[1,0].set_ylabel("Y (cm)")
ax[1,0].xaxis.set_ticks_position('bottom')
ax[1,0].tick_params(top=False, bottom=True)


ax02 = ax[1,1].matshow((pred_init_condition_test).detach()[pick,:,Ns//2,:].T)
plt.colorbar(ax02)
ax[1,1].set_title(" $1^{st}$ test point $I.C._{pred}$ (XZ plane)")
ax[1,1].set_xlabel("X (cm)")
ax[1,1].set_ylabel("Z (cm)")
ax[1,1].xaxis.set_ticks_position('bottom')
ax[1,1].tick_params(top=False, bottom=True)
fig.savefig("figure_1_icbc.pdf")

#2
fig, ax = plt.subplots(2,2, figsize=(15,15))


surf_dt_pred_train = torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz",y_hat_train, bs.Bit_t_derivative, bs.Bit_x, bs.Bit_y, bs.Bit_z)
surf_dt_pred_test = torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz",y_hat_test, bs.Bit_t_derivative, bs.Bit_x, bs.Bit_y, bs.Bit_z)
surf_dxx_pred_train = torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz",y_hat_train, bs.Bit_t, bs.Bit_x_second_derivative, bs.Bit_y, bs.Bit_z)
surf_dxx_pred_test = torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz",y_hat_test, bs.Bit_t, bs.Bit_x_second_derivative, bs.Bit_y, bs.Bit_z)
surf_dxx_pred_train += torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz",y_hat_train, bs.Bit_t, bs.Bit_x,bs.Bit_y_second_derivative,  bs.Bit_z)
surf_dxx_pred_test += torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz",y_hat_test, bs.Bit_t, bs.Bit_x,bs.Bit_y_second_derivative,  bs.Bit_z)
surf_dxx_pred_train += torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz",y_hat_train, bs.Bit_t, bs.Bit_x, bs.Bit_y, bs.Bit_z_second_derivative)
surf_dxx_pred_test += torch.einsum("Nijkl,ti,xj,yk,zl->Ntxyz",y_hat_test, bs.Bit_t, bs.Bit_x, bs.Bit_y, bs.Bit_z_second_derivative)

surf_heat_diff_pred_train = (surf_dt_pred_train - bs.D*surf_dxx_pred_train).detach()
mean_surf_heat_diff_pred_train = torch.mean(surf_heat_diff_pred_train**2, axis = (1,2,3,4))
#std_surf_heat_diff_pred_train = torch.std(surf_heat_diff_pred_train**2, axis = (1,2,3,4))
Q =len(mean_surf_heat_diff_pred_train)
ax[1,0].scatter([1]*Q+np.random.rand(Q)/3, mean_surf_heat_diff_pred_train, label = "Train")
#ax[1,0].errorbar(np.linspace(1,Q,Q), mean_surf_heat_diff_pred_train, std_surf_heat_diff_pred_train)
ax[1,0].set_ylim(-0.001,0.001)

surf_heat_diff_pred_test = (surf_dt_pred_test - bs.D*surf_dxx_pred_test).detach()
mean_surf_heat_diff_pred_test = torch.mean(surf_heat_diff_pred_test**2, axis = (1,2,3,4))
#std_surf_heat_diff_pred_test = torch.std(surf_heat_diff_pred_test**2, axis = (1,2,3,4))
Q =len(mean_surf_heat_diff_pred_test)
ax[1,0].scatter([2]*Q+np.random.rand(Q)/3, mean_surf_heat_diff_pred_test, label = "Test")
#ax[1,0].errorbar(np.linspace(1,Q,Q), mean_surf_heat_diff_pred_test, std_surf_heat_diff_pred_test)
ax[1,0].xaxis.set_ticks_position('bottom')
ax[1,0].tick_params(top=False, bottom=True)
ax[1,0].set_title("mean squared residual over all training and test inputs")
ax[1,0].set_ylim(0.00001,0.0004)
ax[1,0].set_xlim(0.5,2.5)
ax[1,0].set_xticks([1,2])
ax[1,0].set_xticklabels(["Training Data","Test Data"])


# Adjust the position of ax[1, 0] to be half the width
pos1 = ax[1, 0].get_position()  # Get the original position
new_pos1 = [pos1.x0, pos1.y0, pos1.width /1.5, pos1.height]  # Modify the width
ax[1, 0].set_position(new_pos1)  # Set the new position

aa = ax[0,0].matshow(surf_heat_diff_pred_test[pick,-1,:,:,Ns//2].T)
plt.colorbar(aa)
ax[0,0].set_title(r"$U(t=1\,s,x,y,z=0.5\,cm),\;\partial_t U - D\cdot \Delta U$ X-Y slice")
ax[0,0].set_ylabel("Y (cm)")
ax[0,0].set_xlabel("X (cm)")
ax[0,0].set_xticks([i*2 for i in range(Ns//2+1)])
ax[0,0].set_yticks([i*2 for i in range(Ns//2+1)])
ax[0,0].set_xticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[0,0].set_yticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[0,0].xaxis.set_ticks_position('bottom')
ax[0,0].tick_params(top=False, bottom=True)

ab = ax[0,1].matshow(surf_heat_diff_pred_test[pick,:,:,Ns//2,Ns//2])
ax[0,1].set_title(r"$U(t,x\,,y=0.5\,cm,z=0.5\,cm),\;\partial_t U - D\cdot \Delta U$  T-X slice")
ax[0,1].set_xlabel("X (cm)")
ax[0,1].set_ylabel("Time (s)")
ax[0,1].set_xticks([i*2 for i in range(Ns//2+1)])
ax[0,1].set_yticks([i*2 for i in range(Ns//2+1)])
ax[0,1].set_xticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[0,1].set_yticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[0,1].xaxis.set_ticks_position('bottom')
ax[0,1].tick_params(top=False, bottom=True)
plt.colorbar(ab)

print(torch.sum(y_hat_test[:,:,:,:,[0,-1]]).detach())
pred_full_time_test = bs.make_surface(y_hat_test).detach().numpy()

print(np.sum(pred_full_time_test[:,:,:,:,[0,-1]]**2))
fig.savefig("figure_2_heat_eq.pdf")
#3
fig, ax = plt.subplots(1,3, figsize=(21,4))
a = ax[0].matshow(pred_full_time_test[0,:,:,Ns//3,Ns//3])
ax[0].set_title("<U(t,x,y,z)>$_{xz}$ T-X profile")
ax[0].set_xlabel("X (cm)")
ax[0].set_ylabel("T")
ax[0].set_xticks([i*3 for i in range(Ns//3+1)])
ax[0].set_yticks([i*2 for i in range(Ns//2+1)])
ax[0].set_xticklabels([f"{i*3/Ns:0.2f}" for i in range(Ns//3+1)])
ax[0].set_yticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[0].xaxis.set_ticks_position('bottom')
ax[0].tick_params(top=False, bottom=True)
plt.colorbar(a)

b = ax[1].matshow(np.mean(pred_full_time_test[pick], axis = (1,3)))
ax[1].set_title("<U(t,x,y,z)>$_{xz}$ T-Y profile")
ax[1].set_xlabel("Y (cm)")
ax[1].set_ylabel("T")
ax[1].set_xticks([i*3 for i in range(Ns//3+1)])
ax[1].set_yticks([i*2 for i in range(Ns//2+1)])
ax[1].set_xticklabels([f"{i*3/Ns:0.2f}" for i in range(Ns//3+1)])
ax[1].set_yticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[1].xaxis.set_ticks_position('bottom')
ax[1].tick_params(top=False, bottom=True)
plt.colorbar(b)


c = ax[2].matshow(np.mean(pred_full_time_test[pick], axis = (1,2)))
ax[2].set_title("<U(t,x,y,z)>$_{xy}$ T-Z profile")
ax[2].set_xlabel("Z (cm)")
ax[2].set_ylabel("T")
ax[2].set_xticks([i*3 for i in range(Ns//3+1)])
ax[2].set_yticks([i*2 for i in range(Ns//2+1)])
ax[2].set_xticklabels([f"{i*3/Ns:0.2f}" for i in range(Ns//3+1)])
ax[2].set_yticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])

ax[2].xaxis.set_ticks_position('bottom')
ax[2].tick_params(top=False, bottom=True)
plt.colorbar(c)
fig.savefig("figure_3_profiles.pdf")

#4
#ax[1,1].colorbar()
fig, ax = plt.subplots(1,2 , figsize = (15,6))
g = ax[0].matshow(pred_full_time_test[pick,1,:,Ns//2,:])
h = ax[1].matshow(pred_full_time_test[pick,1,:,:,Ns//2])
ax[0].set_title(r"U(t=0.05s,x,y=$\frac{1}{2}$,z) X-Z slice")
ax[1].set_title(r"U(t=0.05s,x,y,z=$\frac{1}{2}$) X-Y slice")

ax[0].set_xlabel("X (cm)")
ax[0].set_ylabel("Z (cm)")
ax[1].set_xlabel("X (cm)")
ax[1].set_ylabel("Y (cm)")

ax[0].set_xticks([i*2 for i in range(Ns//2+1)])
ax[0].set_yticks([i*2 for i in range(Ns//2+1)])
ax[1].set_xticks([i*2 for i in range(Ns//2+1)])
ax[1].set_yticks([i*2 for i in range(Ns//2+1)])
ax[0].set_xticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[0].set_yticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[0].xaxis.set_ticks_position('bottom')
ax[0].tick_params(top=False, bottom=True)
ax[1].set_xticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[1].set_yticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[1].xaxis.set_ticks_position('bottom')
ax[1].tick_params(top=False, bottom=True)
ax[1].xaxis.tick_bottom()
plt.colorbar(g)
plt.colorbar(h)

fig.savefig("figure_4_slices_t005.pdf")

#5
fig, ax = plt.subplots(1,2, figsize = (15,6))
k = ax[0].matshow(pred_full_time_test[pick,-1,:,Ns//2,:])
l = ax[1].matshow(pred_full_time_test[pick,-1,:,:,Ns//2])
ax[0].set_title(r"U(t=1 s,x,y=$\frac{1}{2}$,z) X-Z slice")
ax[0].set_xlabel("X (cm)")
ax[0].set_ylabel("Z (cm)")
ax[0].set_xticks([i*2 for i in range(Ns//2+1)])
ax[0].set_yticks([i*2 for i in range(Ns//2+1)])
ax[0].set_xticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[0].set_yticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[0].xaxis.set_ticks_position('bottom')
ax[0].tick_params(top=False, bottom=True)

ax[1].set_title(r"U(t=1 s,x,y,z=$\frac{1}{2}$) X-Y slice")
ax[1].set_xlabel("X (cm)")
ax[1].set_ylabel("Y (cm)")
ax[1].set_xticks([i*2 for i in range(Ns//2+1)])
ax[1].set_yticks([i*2 for i in range(Ns//2+1)])
ax[1].set_xticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[1].set_yticklabels([f"{i*2/Ns:0.2f}" for i in range(Ns//2+1)])
ax[1].xaxis.set_ticks_position('bottom')
ax[1].tick_params(top=False, bottom=True)
plt.colorbar(k)
plt.colorbar(l)
fig.savefig("figure_5_slices_t095.pdf")

plt.show()