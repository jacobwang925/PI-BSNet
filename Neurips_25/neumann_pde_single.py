import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# 0) Config
# =========================
save_dir = "Neumann_Parametric_Diffusion_Compare"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# PDE: s_t = nu s_xx on x in [0,1], t in [0,T]
T_max = 1.0
x0, x1 = 0.0, 1.0

# Training data
num_train = 50
num_test  = 10
nu_range = (0.1, 1.5)

Nx = 128
Nt = 128
x_np = np.linspace(x0, x1, Nx, dtype=np.float32)
t_np = np.linspace(0.0, T_max, Nt, dtype=np.float32)
Xg, Tg = np.meshgrid(x_np, t_np, indexing="xy")  # shapes [Nt,Nx]

# Loss weights
PDE_WEIGHT  = 1.0
DATA_WEIGHT = 1.0
IC_WEIGHT   = 2.0
BC_WEIGHT   = 2.0

epochs = 5000
lr = 1e-3

# =========================
# 1) Ground truth (closed form)
# =========================
def exact_solution_np(x_grid, t_grid, nu):
    # s(x,t;nu) = cos(pi x) * exp(-nu*pi^2*t)
    # x_grid,t_grid can be meshgrids [Nt,Nx]
    return np.cos(np.pi * x_grid) * np.exp(-nu * (np.pi**2) * t_grid)

# =========================
# 2) B-spline utilities (same style as your code)
# =========================
def BsFun(i, d, t, Ln):
    if d == 0:
        return 1.0 if (Ln[i-1] <= t < Ln[i]) else 0.0
    a = 0.0 if (Ln[d+i-1]-Ln[i-1])==0 else (t-Ln[i-1])/(Ln[d+i-1]-Ln[i-1])
    b = 0.0 if (Ln[d+i]  -Ln[i])  ==0 else (Ln[d+i]-t)/(Ln[d+i]-Ln[i])
    return a*BsFun(i, d-1, t, Ln) + b*BsFun(i+1, d-1, t, Ln)

def BsFun_derivative(i, d, t, Ln):
    if d == 0:
        return 0.0
    a = 0.0 if (Ln[d+i-1]-Ln[i-1])==0 else d/(Ln[d+i-1]-Ln[i-1])
    b = 0.0 if (Ln[d+i]  -Ln[i])  ==0 else d/(Ln[d+i]  -Ln[i])
    return a*BsFun(i, d-1, t, Ln) - b*BsFun(i+1, d-1, t, Ln)

def BsFun_second_derivative(i, d, t, Ln):
    # simple recursion using first-derivative formula twice is expensive;
    # keep your closed-form-like approximation used earlier:
    if d < 2:
        return 0.0
    a = 0.0 if (Ln[d+i-2]-Ln[i-2])==0 else d*(d-1)/((Ln[d+i-2]-Ln[i-2])**2)
    b = 0.0 if (Ln[d+i-1]-Ln[i-1])==0 else 2*d*(d-1)/((Ln[d+i-1]-Ln[i-1])**2)
    c = 0.0 if (Ln[d+i]  -Ln[i])  ==0 else d*(d-1)/((Ln[d+i]  -Ln[i])**2)
    return a*BsFun(i, d-2, t, Ln) - b*BsFun(i+1, d-2, t, Ln) + c*BsFun(i+2, d-2, t, Ln)

def BsKnots(n_cp, d, Ns):
    n_knots = n_cp + d + 1
    Ln = np.zeros(n_knots, dtype=np.float32)
    for i in range(d+1, n_knots-d-1):
        Ln[i] = i - d
    Ln[n_knots-d-1:] = n_cp - d
    tk = np.linspace(0, Ln[-1], Ns).astype(np.float32)

    B = np.zeros((Ns, n_cp), dtype=np.float32)
    for j in range(n_cp):
        for i in range(Ns):
            B[i, j] = BsFun(j+1, d, tk[i], Ln)
    B[-1, -1] = 1.0
    return tk, Ln, B

def BsKnots_derivatives(n_cp, d, Ns, Ln, tk):
    Bd1 = np.zeros((Ns, n_cp), dtype=np.float32)
    Bd2 = np.zeros((Ns, n_cp), dtype=np.float32)
    for j in range(n_cp):
        for i in range(Ns):
            Bd1[i, j] = BsFun_derivative(j+1, d, tk[i], Ln)
            Bd2[i, j] = BsFun_second_derivative(j+1, d, tk[i], Ln)
    return Bd1, Bd2

def bspline_eval(U, Bt, Bx):
    # Bt: [Nt, n_cp_t], Bx: [Nx, n_cp_x], U: [n_cp_t, n_cp_x]
    # returns S: [Nt, Nx]
    return Bt @ U @ Bx.T

def bspline_derivs(U, Bt, Bx, Bt_d1, Bx_d1, Bt_d2, Bx_d2):
    S_t  = Bt_d1 @ U @ Bx.T
    S_x  = Bt    @ U @ Bx_d1.T
    S_xx = Bt    @ U @ Bx_d2.T
    return S_t, S_x, S_xx

# =========================
# 3) Build datasets (train/test)
# =========================
def make_dataset(num_samples):
    data = []
    for _ in range(num_samples):
        nu = np.random.uniform(*nu_range)
        S_true = exact_solution_np(Xg, Tg, nu)  # [Nt,Nx]
        data.append({
            "nu": torch.tensor([[nu]], dtype=torch.float32),
            "S_true": torch.tensor(S_true, dtype=torch.float32),  # [Nt,Nx]
        })
    return data

train_data = make_dataset(num_train)
test_data  = make_dataset(num_test)

# =========================
# 4) Models
# =========================
class VanillaPINN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t, nu):
        # x,t: [N,1], nu: [N,1]
        inp = torch.cat([x, t, nu], dim=1)
        return self.net(inp)

class PIDeepONet(nn.Module):
    def __init__(self, branch_hidden=128, trunk_hidden=128):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(1, branch_hidden), nn.Tanh(),
            nn.Linear(branch_hidden, branch_hidden), nn.Tanh(),
            nn.Linear(branch_hidden, trunk_hidden), nn.Tanh(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, trunk_hidden), nn.Tanh(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.Tanh(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.Tanh(),
        )

    def forward(self, x, t, nu):
        # nu is [N,1], but branch expects [batch,1]
        b = self.branch(nu)                  # [N, trunk_hidden]
        tr = self.trunk(torch.cat([x,t], 1)) # [N, trunk_hidden]
        return torch.sum(b * tr, dim=1, keepdim=True)

class ControlPointNet(nn.Module):
    def __init__(self, n_cp_t, n_cp_x, hidden=128):
        super().__init__()
        self.n_cp_t, self.n_cp_x = n_cp_t, n_cp_x
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_cp_t * n_cp_x)
        )

    def forward(self, nu):
        # nu: [1,1] or [B,1] -> output [B, n_cp_t*n_cp_x]
        out = self.net(nu)
        return out

class BSNetLoss(nn.Module):
    """
    B-spline surface s(x,t) = Bt U Bx^T, where U is predicted from nu.
    Enforces PDE + IC + Neumann BC via loss penalties (no hard assignment).
    """
    def __init__(self, n_cp_t, n_cp_x, hidden=128):
        super().__init__()
        self.n_cp_t, self.n_cp_x = n_cp_t, n_cp_x
        self.ctrl = ControlPointNet(n_cp_t, n_cp_x, hidden)

    def forward_U(self, nu):
        vec = self.ctrl(nu)  # [B, n_cp_t*n_cp_x]
        # for simplicity B=1 in our loops
        U = vec.view(-1, self.n_cp_t, self.n_cp_x)  # [B,n_cp_t,n_cp_x]
        return U

# =========================
# 5) Precompute spline bases on uniform grids x,t in [0,1]x[0,T]
# =========================
# choose spline resolution (# control points) modestly
n_cp_x, n_cp_t, d = 25, 25, 3

# Knot param for x and t: we map x,t to the knot parameter domain [0, Ln[-1]]
tk_x, Ln_x, Bx_np = BsKnots(n_cp_x, d, Nx)
Bx_d1_np, Bx_d2_np = BsKnots_derivatives(n_cp_x, d, Nx, Ln_x, tk_x)

tk_t, Ln_t, Bt_np = BsKnots(n_cp_t, d, Nt)
Bt_d1_np, Bt_d2_np = BsKnots_derivatives(n_cp_t, d, Nt, Ln_t, tk_t)

Bx     = torch.tensor(Bx_np, dtype=torch.float32, device=device)      # [Nx,n_cp_x]
Bx_d1  = torch.tensor(Bx_d1_np, dtype=torch.float32, device=device)
Bx_d2  = torch.tensor(Bx_d2_np, dtype=torch.float32, device=device)
Bt     = torch.tensor(Bt_np, dtype=torch.float32, device=device)      # [Nt,n_cp_t]
Bt_d1  = torch.tensor(Bt_d1_np, dtype=torch.float32, device=device)
Bt_d2  = torch.tensor(Bt_d2_np, dtype=torch.float32, device=device)

# Index helpers for boundary columns in x-grid (x=0 and x=1)
idx_x0 = 0
idx_x1 = Nx - 1

# Ground-truth IC on the grid
ic_true = torch.tensor(np.cos(np.pi * x_np), dtype=torch.float32, device=device)  # [Nx]

# =========================
# 6) Loss helpers
# =========================
def pinn_pde_residual(u, x, t, nu):
    # u: [N,1], x,t,nu require_grad
    u_t  = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x  = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - nu * u_xx

def relative_L2(pred, true, eps=1e-12):
    # pred,true: tensors [Nt,Nx]
    num = torch.sqrt(torch.mean((pred-true)**2))
    den = torch.sqrt(torch.mean(true**2)) + eps
    return (num/den).item()

# =========================
# 7) Train loop (3 models only)
# =========================
models = {
    "PINN": VanillaPINN(hidden=128).to(device),
    "PI-DeepONet": PIDeepONet(branch_hidden=128, trunk_hidden=128).to(device),
    "BSNet-loss": BSNetLoss(n_cp_t=n_cp_t, n_cp_x=n_cp_x, hidden=128).to(device),
}

optims = {k: optim.Adam(m.parameters(), lr=lr) for k,m in models.items()}

history = {k: {"total": [], "pde": [], "data": [], "ic": [], "bc": []} for k in models.keys()}
train_time = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    start = time.time()

    optimizer = optims[name]
    model.train()

    for ep in tqdm(range(1, epochs+1), desc=name):
        optimizer.zero_grad()

        total_L = 0.0
        Lpde_acc = 0.0
        Ldata_acc = 0.0
        Lic_acc = 0.0
        Lbc_acc = 0.0

        for sample in train_data:
            nu = sample["nu"].to(device)        # [1,1]
            S_true = sample["S_true"].to(device) # [Nt,Nx]

            if name in ["PINN", "PI-DeepONet"]:
                # Flatten grid
                xf = torch.tensor(Xg.reshape(-1,1), dtype=torch.float32, device=device, requires_grad=True)
                tf = torch.tensor(Tg.reshape(-1,1), dtype=torch.float32, device=device, requires_grad=True)
                nuf = nu.expand_as(xf).clone().detach().requires_grad_(True)

                u_pred = model(xf, tf, nuf)  # [Nt*Nx,1]
                u_true = S_true.reshape(-1,1)

                # PDE
                res = pinn_pde_residual(u_pred, xf, tf, nuf)
                Lpde = torch.mean(res**2)

                # Data
                Ldata = torch.mean((u_pred - u_true)**2)

                # IC: t=0 slice
                # pick indices where t=0 -> first Nt block in meshgrid ordering "xy"
                # Safer: build explicit IC points
                x_ic = torch.tensor(x_np.reshape(-1,1), dtype=torch.float32, device=device, requires_grad=True)
                t_ic = torch.zeros_like(x_ic, device=device, requires_grad=True)
                nu_ic = nu.expand_as(x_ic).clone().detach().requires_grad_(True)
                u_ic_pred = model(x_ic, t_ic, nu_ic)  # [Nx,1]
                u_ic_true = torch.cos(np.pi * x_ic)
                Lic = torch.mean((u_ic_pred - u_ic_true)**2)

                # Neumann BC: u_x(0,t)=0 and u_x(1,t)=0
                t_bc = torch.tensor(t_np.reshape(-1,1), dtype=torch.float32, device=device, requires_grad=True)
                x0_bc = torch.zeros_like(t_bc, device=device, requires_grad=True)
                x1_bc = torch.ones_like(t_bc, device=device, requires_grad=True)
                nu_bc0 = nu.expand_as(t_bc).clone().detach().requires_grad_(True)
                nu_bc1 = nu.expand_as(t_bc).clone().detach().requires_grad_(True)

                u0 = model(x0_bc, t_bc, nu_bc0)
                u1 = model(x1_bc, t_bc, nu_bc1)
                ux0 = autograd.grad(u0, x0_bc, grad_outputs=torch.ones_like(u0), create_graph=True)[0]
                ux1 = autograd.grad(u1, x1_bc, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
                Lbc = torch.mean(ux0**2) + torch.mean(ux1**2)

            else:
                # BSNet-loss: evaluate on full grid via spline bases
                U = model.forward_U(nu)[0]  # [n_cp_t, n_cp_x]

                S_pred = bspline_eval(U, Bt, Bx)  # [Nt,Nx]
                S_t, S_x, S_xx = bspline_derivs(U, Bt, Bx, Bt_d1, Bx_d1, Bt_d2, Bx_d2)

                # PDE residual: S_t - nu S_xx = 0
                Lpde = torch.mean((S_t - nu.item() * S_xx)**2)

                # Data loss
                Ldata = torch.mean((S_pred - S_true)**2)

                # IC loss: at t=0 => first row in t-grid
                Lic = torch.mean((S_pred[0, :] - ic_true)**2)

                # Neumann BC loss: S_x(x=0,t)=0 and S_x(x=1,t)=0
                # S_x is [Nt,Nx]
                Lbc = torch.mean(S_x[:, idx_x0]**2) + torch.mean(S_x[:, idx_x1]**2)

            L = PDE_WEIGHT*Lpde + DATA_WEIGHT*Ldata + IC_WEIGHT*Lic + BC_WEIGHT*Lbc
            total_L = total_L + L

            Lpde_acc += Lpde.item()
            Ldata_acc += Ldata.item()
            Lic_acc += Lic.item()
            Lbc_acc += Lbc.item()

        total_L.backward()
        optimizer.step()

        history[name]["total"].append(total_L.item())
        history[name]["pde"].append(Lpde_acc)
        history[name]["data"].append(Ldata_acc)
        history[name]["ic"].append(Lic_acc)
        history[name]["bc"].append(Lbc_acc)

    elapsed = time.time() - start
    train_time[name] = elapsed
    torch.save(model.state_dict(), os.path.join(save_dir, f"{name}_ckpt.pth"))
    print(f"Saved {name} in {elapsed:.1f}s")

# =========================
# 8) Testing + visualization
# =========================
def eval_model_on_sample(model_name, model, nu, S_true):
    model.eval()
    with torch.no_grad():
        if model_name == "BSNet-loss":
            U = model.forward_U(nu.to(device))[0]
            S_pred = bspline_eval(U, Bt, Bx)
        else:
            xf = torch.tensor(Xg.reshape(-1,1), dtype=torch.float32, device=device)
            tf = torch.tensor(Tg.reshape(-1,1), dtype=torch.float32, device=device)
            nuf = nu.to(device).expand_as(xf)
            u_pred = model(xf, tf, nuf).reshape(Nt, Nx)
            S_pred = u_pred
    rel = relative_L2(S_pred, S_true.to(device))
    return S_pred.detach().cpu().numpy(), rel

# pick one test case to plot
nu_plot = test_data[0]["nu"]
S_true_plot = test_data[0]["S_true"]
errs = {}

for name, model in models.items():
    pred, rel = eval_model_on_sample(name, model, nu_plot, S_true_plot)
    errs[name] = rel
    print(f"[Test] {name}: rel-L2 = {rel:.3e}")

    # heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    im0 = axes[0].imshow(S_true_plot.cpu().numpy(), origin="lower", aspect="auto",
                         extent=[x0,x1,0,T_max])
    axes[0].set_title("True")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t"); plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred, origin="lower", aspect="auto",
                         extent=[x0,x1,0,T_max])
    axes[1].set_title(f"{name} Pred")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("t"); plt.colorbar(im1, ax=axes[1])

    err = np.abs(pred - S_true_plot.cpu().numpy())
    im2 = axes[2].imshow(err, origin="lower", aspect="auto",
                         extent=[x0,x1,0,T_max])
    axes[2].set_title(f"Abs Error (rel-L2={rel:.2e})")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("t"); plt.colorbar(im2, ax=axes[2])

    fig.suptitle(f"nu={nu_plot.item():.3f}")
    plt.show()
    fig.savefig(os.path.join(save_dir, f"heatmap_{name}.png"), dpi=200)

# plot training curves
plt.figure(figsize=(10,6))
for name in models.keys():
    plt.plot(history[name]["total"], label=name)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Total loss (log)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "train_total_loss.png"), dpi=200)
plt.show()


# tradeoff plot: train time vs test error (avg over test set)
avg_err = {}
for name, model in models.items():
    rels = []
    for sample in test_data:
        _, rel = eval_model_on_sample(name, model, sample["nu"], sample["S_true"])
        rels.append(rel)
    avg_err[name] = float(np.mean(rels))
    print(f"[Avg Test] {name}: rel-L2 = {avg_err[name]:.3e}")

plt.figure(figsize=(7,5))
for name in models.keys():
    plt.scatter(train_time[name], avg_err[name], s=100)
    plt.text(train_time[name]*1.05, avg_err[name]*1.05, name)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Training time (s)")
plt.ylabel("Avg test relative L2 error")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "tradeoff.png"), dpi=200)
plt.show()

print("Done. Plots saved to:", save_dir)