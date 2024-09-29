import torch
import matplotlib.pyplot as plt
from modules import ControlPointNet3D
from base_run import base_run
from scripts.bs_net_fokker_planck import bsnet_train

a = torch.tensor([[0,0,0,0.3]],dtype=torch.float32)
filename = "models/3dfp/v0/20240929_121110_loss_0.747239.pt"
data = torch.load(filename)

bs = bsnet_train()
bs.setup("scripts/config_fp.json")

torch.load(filename)
bs.model.load_state_dict(data["model_state_dict"])
print(a[:,:-1])
init_condition = bs.gaussian_pdf(a[0,:-1], a[0,-1:])

y_hat = bs.model(a)
Ns = bs.n_points
ncpx = bs.n_ctrl_pts_state
ncpt = bs.n_ctrl_pts_time
z = torch.vstack([init_condition.reshape(1,ncpx,ncpx,ncpx), 
                  y_hat.reshape(ncpt-1,ncpx,ncpx,ncpx)]
                 ).reshape(1,ncpt,ncpx,ncpx,ncpx)

surf = bs.make_surface(z).detach().numpy()
surf0 =surf[0]
plt.matshow(surf0[:,Ns//2,Ns//2,:])
plt.show()