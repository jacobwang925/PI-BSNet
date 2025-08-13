# ood_test.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from neuralop.models.fno import TFNO2d  # pip install neuraloperator

# ─── 1) B‑SPLINE & GROUND‑TRUTH HELPERS ────────────────────────────────────────

def BsFun(i, d, t, Ln):
    if d == 0:
        return 1.0 if Ln[i-1] <= t < Ln[i] else 0.0
    a = 0 if (Ln[d+i-1]-Ln[i-1])==0 else (t-Ln[i-1])/(Ln[d+i-1]-Ln[i-1])
    b = 0 if (Ln[d+i]  -Ln[i])  ==0 else (Ln[d+i]-t)/(Ln[d+i]-Ln[i])
    return a*BsFun(i,   d-1, t, Ln) + b*BsFun(i+1, d-1, t, Ln)

def BsFun_derivative(i, d, t, Ln):
    if d == 0: return 0.0
    a = 0 if (Ln[d+i-1]-Ln[i-1])==0 else d/(Ln[d+i-1]-Ln[i-1])
    b = 0 if (Ln[d+i]  -Ln[i])  ==0 else d/(Ln[d+i]  -Ln[i])
    return a*BsFun(i,   d-1, t, Ln) - b*BsFun(i+1, d-1, t, Ln)

def BsFun_second_derivative(i, d, t, Ln):
    if d < 2: return 0.0
    a = 0 if (Ln[d+i-2]-Ln[i-2])==0 else d*(d-1)/((Ln[d+i-2]-Ln[i-2])**2)
    b = 0 if (Ln[d+i-1]-Ln[i-1])==0 else 2*d*(d-1)/((Ln[d+i-1]-Ln[i-1])**2)
    c = 0 if (Ln[d+i]  -Ln[i])  ==0 else d*(d-1)/((Ln[d+i]  -Ln[i])  **2)
    return a*BsFun(i,   d-2, t, Ln) - b*BsFun(i+1, d-2, t, Ln) + c*BsFun(i+2, d-2, t, Ln)

def BsKnots(n_cp, d, Ns):
    n_knots = n_cp + d + 1
    Ln = np.zeros(n_knots)
    for i in range(d+1, n_knots-d-1):
        Ln[i] = i - d
    Ln[n_knots-d-1:] = n_cp - d
    tk = np.linspace(0, Ln[-1], Ns)
    Bit = np.zeros((Ns, n_cp))
    for j in range(n_cp):
        for i in range(Ns):
            Bit[i,j] = BsFun(j+1, d, tk[i], Ln)
    Bit[-1,-1] = 1.0
    return tk, Ln, Bit

def ground_truth(x_vals, T_vals, a, lam):
    def f(x,t):
        if t==0: return 0.0
        return (a-x)/np.sqrt(2*np.pi*t**3) * np.exp(-((a-x)-lam*t)**2/(2*t))
    def F2(x,T):
        return 1.0 if x>=a else quad(lambda tt: f(x,tt), 0, T)[0]
    F = np.zeros((len(T_vals), len(x_vals)))
    for i,x in enumerate(x_vals):
        for j,Tj in enumerate(T_vals):
            F[j,i] = F2(x,Tj)
    return F

# ─── 2) MODEL CLASSES ─────────────────────────────────────────────────────────

import torch.nn as nn
import torch.autograd as autograd

class EPINN(nn.Module):
    def __init__(self, x_min, T, hidden=64):
        super().__init__()
        self.x_min, self.T = x_min, T
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x, t, lam, a):
        phi_x = (a - x) / (a - self.x_min)
        phi_t = t / self.T
        denom = (phi_t + phi_x - phi_t * phi_x).clamp(min=1e-6)
        g = phi_t / denom
        inp = torch.cat([x, t, lam.expand_as(x), a.expand_as(x)], dim=1)
        return g + phi_x * phi_t * self.net(inp)

class SFHCPINN(nn.Module):
    def __init__(self, x_min, T, hidden=64, nf=20):
        super().__init__()
        self.x_min, self.T = x_min, T
        self.ff = nn.Linear(2, nf, bias=False)
        nn.init.xavier_normal_(self.ff.weight)
        self.corr = nn.Sequential(
            nn.Linear(2*nf+2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x, t, lam, a):
        phi_x = (a - x) / (a - self.x_min)
        phi_t = t / self.T
        denom = (phi_t + phi_x - phi_t * phi_x).clamp(min=1e-6)
        g = phi_t / denom
        Ff0 = self.ff(torch.cat([x,t],dim=1))
        Ff  = torch.cat([Ff0.sin(), Ff0.cos()], dim=1)
        inp = torch.cat([Ff, lam.expand_as(x), a.expand_as(x)], dim=1)
        return g + phi_x*phi_t * self.corr(inp)

class WideBodyPINN(nn.Module):
    def __init__(self, x_min, T, hidden=64):
        super().__init__()
        self.x_min, self.T = x_min, T
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x, t, lam, a):
        phi_x = (a - x) / (a - self.x_min)
        phi_t = t / self.T
        denom = (phi_t + phi_x - phi_t*phi_x).clamp(min=1e-6)
        g = phi_t / denom
        inp = torch.cat([x, t, lam.expand_as(x), a.expand_as(x)], dim=1)
        return g + phi_x*phi_t * self.net(inp)

class VSPINN(nn.Module):
    def __init__(self, x_min, T, hidden=64):
        super().__init__()
        self.x_min, self.T = x_min, T
        self.sx = nn.Parameter(torch.tensor(1.0))
        self.st = nn.Parameter(torch.tensor(1.0))
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x, t, lam, a):
        xs = (x - self.x_min)*self.sx + self.x_min
        ts = t * self.st
        phi_x = (a - xs) / (a - self.x_min)
        phi_t = ts / self.T
        denom = (phi_t + phi_x - phi_t*phi_x).clamp(min=1e-6)
        g = phi_t / denom
        inp = torch.cat([xs, ts, lam.expand_as(xs), a.expand_as(xs)], dim=1)
        return g + phi_x*phi_t * self.net(inp)

class CANPINN(nn.Module):
    def __init__(self, x_min, T, hidden=64, eps=1e-3):
        super().__init__()
        self.x_min, self.T, self.eps = x_min, T, eps
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x, t, lam, a):
        phi_x = (a - x) / (a - self.x_min)
        phi_t = t / self.T
        denom = (phi_t + phi_x - phi_t*phi_x).clamp(min=1e-6)
        g = phi_t / denom
        inp = torch.cat([x, t, lam.expand_as(x), a.expand_as(x)], dim=1)
        return g + phi_x*phi_t * self.net(inp)
    def pde_residual(self, u, x, t, lam, a):
        u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        up  = self.forward(x+self.eps, t, lam, a)
        um  = self.forward(x-self.eps, t, lam, a)
        u_xx = (up - 2*u + um) / (self.eps**2)
        return u_t - lam*u_x - 0.5*u_xx

class VanillaPINN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x, t, lam, a):
        inp = torch.cat([x, t, lam.expand_as(x), a.expand_as(x)], dim=1)
        return self.net(inp)

class PIDeepONet(nn.Module):
    def __init__(self, branch_hidden=64, trunk_hidden=64):
        super().__init__()
        self.branch_net = nn.Sequential(
            nn.Linear(2, branch_hidden), nn.Tanh(),
            nn.Linear(branch_hidden, branch_hidden), nn.Tanh()
        )
        self.trunk_net  = nn.Sequential(
            nn.Linear(2, trunk_hidden), nn.Tanh(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.Tanh()
        )
        self.fc = nn.Linear(branch_hidden, trunk_hidden, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
    def forward(self, x, t, lam, a):
        b  = self.branch_net(torch.cat([lam, a], dim=1))
        tr = self.trunk_net(torch.cat([x, t],   dim=1))
        return torch.sum(self.fc(b) * tr, dim=1, keepdim=True)

class PIFNO(nn.Module):
    def __init__(self, modes_x=12, modes_t=12, width=32):
        super().__init__()
        self.fno = TFNO2d(modes_x, modes_t, width, in_channels=4, out_channels=1, n_layers=4)
    def forward(self, P):
        return self.fno(P)

class ControlPointNet(nn.Module):
    def __init__(self, n_cp_x, n_cp_t, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, (n_cp_t-1)*(n_cp_x-1))
    def forward(self, lam, a):
        x = torch.cat([lam,a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PIDBSN(nn.Module):
    def __init__(self, n_cp_x, n_cp_t, d, hidden_dim=64):
        super().__init__()
        self.n_cp_x, self.n_cp_t = n_cp_x, n_cp_t
        self.ctrl = ControlPointNet(n_cp_x, n_cp_t, hidden_dim)
    def forward(self, lam, a):
        u      = self.ctrl(lam, a)
        U_pred = u.view(1, self.n_cp_t-1, self.n_cp_x-1)
        return U_pred

# ─── 3) MODEL CONSTRUCTORS ────────────────────────────────────────────────────

x_min, T_max    = -10.0, 10.0
hidden_dim      = 64
n_cp_x, n_cp_t, d = 25, 25, 3

model_constructors = {
    'EPINN':       lambda: EPINN(x_min, T_max, hidden_dim),
    'SFHCPINN':    lambda: SFHCPINN(x_min, T_max, hidden_dim),
    'HWPINN':      lambda: WideBodyPINN(x_min, T_max, hidden_dim),
    'VS-PINN':     lambda: VSPINN(x_min, T_max, hidden_dim),
    'CAN-PINN':    lambda: CANPINN(x_min, T_max, hidden_dim),
    'VanillaPINN': lambda: VanillaPINN(hidden_dim),
    'PIDeepONet':  lambda: PIDeepONet(64, 64),
    'PIFNO':       lambda: PIFNO(12, 12, 32),
    'PIDBSN_FD':   lambda: PIDBSN(n_cp_x, n_cp_t, d, hidden_dim),
    'PIDBSN':      lambda: PIDBSN(n_cp_x, n_cp_t, d, hidden_dim),
}

# ─── 4) OUT‑OF‑DISTRIBUTION TEST LOOP ─────────────────────────────────────────

# save_dir    = 'Baseline_Ablation_Trained_Models'
# epochs      = 5000
# model_names = list(model_constructors.keys())

# # define OOD (λ,a)
# test_params = [(2.5,5.0), (3.0,4.5), (2.2,4.2)]

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # time‑spline tensor for PI‑DBSN
# _, _, Bit_t_np = BsKnots(n_cp_t, d, len(np.linspace(0, T_max, 101)))
# Bt = torch.tensor(Bit_t_np, dtype=torch.float32)

# for name in model_names:
#     model = model_constructors[name]().to(device)
#     ckpt  = os.path.join(save_dir, f"{name}_{epochs}epochs.pth")
#     state = torch.load(ckpt, map_location=device)
#     state.pop('_metadata', None)
#     model.load_state_dict(state)
#     model.eval()

#     for lam_test, a_test in test_params:
#         print(f"\n=== {name} @ λ={lam_test:.2f}, a={a_test:.2f} (OOD) ===")

#         nx, nt    = 101, 101
#         x_test    = np.linspace(x_min, a_test, nx)
#         t_test    = np.linspace(0.0, T_max, nt)
#         Xg, Tg    = np.meshgrid(x_test, t_test, indexing='xy')
#         F_true    = ground_truth(x_test, t_test, a_test, lam_test)

#         with torch.no_grad():
#             if name in ['PIDBSN','PIDBSN_FD']:
#                 _, _, Bit_x_np = BsKnots(n_cp_x, d, nx)
#                 Bx_test = torch.tensor(Bit_x_np, dtype=torch.float32, device=device)

#                 lam_t   = torch.tensor([[lam_test]], dtype=torch.float32, device=device)
#                 a_t     = torch.tensor([[a_test]],   dtype=torch.float32, device=device)
#                 U_pred  = model(lam_t, a_t)[0]
#                 U_full  = torch.ones(n_cp_t, n_cp_x, device=device)
#                 U_full[0,:]   = 0.0
#                 U_full[:,-1]  = 1.0
#                 U_full[1:,:-1]= U_pred
#                 B_pred  = (Bt.to(device) @ U_full @ Bx_test.T).cpu().numpy()

#             elif name == 'PIFNO':
#                 lam_fld = np.full((nt,nx), lam_test, dtype=np.float32)
#                 a_fld   = np.full_like(lam_fld, a_test)
#                 x_fld   = np.tile(x_test[None,:], (nt,1))
#                 t_fld   = np.tile(t_test[:,None], (1,nx))
#                 Ptest   = np.stack([lam_fld,a_fld,x_fld,t_fld], axis=0)
#                 Pbatch  = torch.tensor(Ptest, dtype=torch.float32) \
#                               .unsqueeze(0).permute(0,1,3,2).to(device)
#                 out     = model(Pbatch).cpu().numpy()[0,0]
#                 B_pred  = out.T

#             else:
#                 xf   = torch.tensor(Xg.flatten()[:,None], dtype=torch.float32, device=device)
#                 tf   = torch.tensor(Tg.flatten()[:,None], dtype=torch.float32, device=device)
#                 lam_f= torch.full_like(xf, lam_test)
#                 a_f  = torch.full_like(xf, a_test)
#                 U_flat = model(xf, tf, lam_f, a_f).cpu().numpy()
#                 B_pred = U_flat.reshape(Xg.shape)

#         err = np.abs(B_pred - F_true)
#         L2  = np.linalg.norm(err)/err.size
#         print(f" L2 error = {L2:.3e}, max error = {err.max():.3e}")

#         # optional plot
#         fig, axes = plt.subplots(1,3, figsize=(15,4), subplot_kw={'projection':'3d'})
#         for ax, surf, title in zip(axes, [F_true, B_pred, err], ['True','Pred','Error']):
#             ax.plot_surface(Xg, Tg, surf, edgecolor='none')
#             ax.set_title(f"{name} {title}")
#         plt.tight_layout()
#         plt.show()

# 10 test cases with average L2 error

# ─── 4) OUT‑OF‑DISTRIBUTION TEST LOOP ─────────────────────────────────────────

save_dir    = 'Baseline_Ablation_Trained_Models'
epochs      = 5000
model_names = list(model_constructors.keys())

# define 10 OOD (λ,a) pairs
test_params = [
    (2.5, 5.0),
    (3.0, 4.5),
    (2.2, 4.2),
    (4.0, 6.0),
    (1.5, 3.5),
    (5.0, 5.5),
    (0.5, 2.0),
    (3.5, 7.0),
    (2.0, 8.0),
    (1.0, 6.0),
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# time‑spline tensor for PI‑DBSN
_, _, Bit_t_np = BsKnots(n_cp_t, d, len(np.linspace(0, T_max, 101)))
Bt = torch.tensor(Bit_t_np, dtype=torch.float32, device=device)

# accumulate L2 error
total_L2 = {name: 0.0 for name in model_names}

for name in model_names:
    # load trained model
    model = model_constructors[name]().to(device)
    ckpt  = os.path.join(save_dir, f"{name}_{epochs}epochs.pth")
    state = torch.load(ckpt, map_location=device)
    state.pop('_metadata', None)
    model.load_state_dict(state)
    model.eval()

    for lam_test, a_test in test_params:
        print(f"\n=== {name} @ λ={lam_test:.2f}, a={a_test:.2f} (OOD) ===")

        # prepare test grid and ground truth
        nx, nt    = 101, 101
        x_test    = np.linspace(x_min, a_test, nx)
        t_test    = np.linspace(0.0, T_max, nt)
        Xg, Tg    = np.meshgrid(x_test, t_test, indexing='xy')
        F_true    = ground_truth(x_test, t_test, a_test, lam_test)

        with torch.no_grad():
            # generate prediction B_pred according to model type
            if name in ['PIDBSN','PIDBSN_FD']:
                _, _, Bit_x_np = BsKnots(n_cp_x, d, nx)
                Bx_test = torch.tensor(Bit_x_np, dtype=torch.float32, device=device)

                lam_t  = torch.tensor([[lam_test]], dtype=torch.float32, device=device)
                a_t    = torch.tensor([[a_test]],   dtype=torch.float32, device=device)
                U_pred = model(lam_t, a_t)[0]

                # reconstruct full U and evaluate spline
                U_full = torch.ones(n_cp_t, n_cp_x, device=device)
                U_full[0,:]   = 0.0
                U_full[:,-1]  = 1.0
                U_full[1:,:-1]= U_pred
                B_pred = (Bt @ U_full @ Bx_test.T).cpu().numpy()

            elif name == 'PIFNO':
                lam_fld = np.full((nt,nx), lam_test, dtype=np.float32)
                a_fld   = np.full_like(lam_fld, a_test)
                x_fld   = np.tile(x_test[None,:], (nt,1))
                t_fld   = np.tile(t_test[:,None], (1,nx))
                Ptest   = np.stack([lam_fld,a_fld,x_fld,t_fld], axis=0)
                Pbatch  = torch.tensor(Ptest, dtype=torch.float32).unsqueeze(0).permute(0,1,3,2).to(device)
                out     = model(Pbatch).cpu().numpy()[0,0]
                B_pred  = out.T

            else:
                xf    = torch.tensor(Xg.flatten()[:,None], dtype=torch.float32, device=device)
                tf    = torch.tensor(Tg.flatten()[:,None], dtype=torch.float32, device=device)
                lam_f = torch.full_like(xf, lam_test)
                a_f   = torch.full_like(xf, a_test)
                U_flat = model(xf, tf, lam_f, a_f).cpu().numpy()
                B_pred = U_flat.reshape(Xg.shape)

        # compute errors
        err = np.abs(B_pred - F_true)
        L2  = np.linalg.norm(err) / err.size
        total_L2[name] += L2
        print(f" L2 error = {L2:.3e}, max error = {err.max():.3e}")

# print average L2 over all OOD combinations
print("\n=== Average L2 Error Across All OOD Params ===")
for name in model_names:
    avg_L2 = total_L2[name] / len(test_params)
    print(f"{name:12s} : {avg_L2:.3e}")
