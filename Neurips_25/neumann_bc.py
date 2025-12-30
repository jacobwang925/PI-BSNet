import os, time, json, pickle, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# 0) Config
# ============================================================
save_dir = "Neumann_Parametric_Diffusion_Compare"
os.makedirs(save_dir, exist_ok=True)

# --------- Reproducibility ----------
SEED = 42
DETERMINISTIC_CUDA = True   # True => try strict-ish determinism (see note below)

# --------- Load vs Train switches ----------
# True  => retrain this model now
# False => load checkpoint if available, otherwise train
TRAIN_MODELS = {
    "PINN": False,
    "PI-DeepONet": False,
    "PI-BSNet": False,
}

# --------- Dataset persistence ----------
SAVE_DATASETS = True
LOAD_DATASETS_IF_EXIST = True
dataset_path = os.path.join(save_dir, f"dataset_seed{SEED}.pkl")

# --------- Device ----------
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
Xg, Tg = np.meshgrid(x_np, t_np, indexing="xy")  # [Nt,Nx]

# Loss weights
PDE_WEIGHT  = 1.0
DATA_WEIGHT = 5.0
IC_WEIGHT   = 2.0
BC_WEIGHT   = 2.0

epochs = 5000
lr = 1e-3

# --------- B-spline hyperparams (you'll play with these) ----------
BSPLINE_CFG = {
    "n_cp_x": 40,
    "n_cp_t": 40,
    "degree": 5,
}

# ============================================================
# 0.1) Seeding utils
# ============================================================
def set_all_seeds(seed: int, deterministic_cuda: bool = True):
    """
    Note:
    - Setting torch.use_deterministic_algorithms(True) on CUDA can error unless you set
      CUBLAS_WORKSPACE_CONFIG before launching python.
    - Here we do "best effort": seed everything + cudnn flags; we do NOT force strict
      deterministic algorithms to avoid runtime crash.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Do NOT force strict deterministic algorithms here to avoid CuBLAS errors.
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

set_all_seeds(SEED, DETERMINISTIC_CUDA)

# ============================================================
# 0.2) File naming (avoid overwrite for BSNet sweeps)
# ============================================================
def model_tag(model_name: str) -> str:
    if model_name == "PI-BSNet":
        return f"{model_name}_cp{BSPLINE_CFG['n_cp_t']}x{BSPLINE_CFG['n_cp_x']}_d{BSPLINE_CFG['degree']}_seed{SEED}"
    else:
        return f"{model_name}_seed{SEED}"

def ckpt_paths(model_name: str):
    tag = model_tag(model_name)
    ckpt = os.path.join(save_dir, f"{tag}_ckpt.pth")
    opt  = os.path.join(save_dir, f"{tag}_optim.pth")
    meta = os.path.join(save_dir, f"{tag}_meta.json")
    return ckpt, opt, meta

# ============================================================
# 1) Ground truth (closed form)
# ============================================================
def exact_solution_np(x_grid, t_grid, nu):
    # s(x,t;nu) = cos(pi x) * exp(-nu*pi^2*t)
    return np.cos(np.pi * x_grid) * np.exp(-nu * (np.pi**2) * t_grid)

# ============================================================
# 2) B-spline utilities (same style as your code)
# ============================================================
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
    return Bt @ U @ Bx.T

def bspline_derivs(U, Bt, Bx, Bt_d1, Bx_d1, Bt_d2, Bx_d2):
    S_t  = Bt_d1 @ U @ Bx.T
    S_x  = Bt    @ U @ Bx_d1.T
    S_xx = Bt    @ U @ Bx_d2.T
    return S_t, S_x, S_xx

# ============================================================
# 3) Build datasets (train/test) with option to persist
# ============================================================
def make_dataset(num_samples, nu_rng, Xg, Tg):
    data = []
    for _ in range(num_samples):
        nu = np.random.uniform(*nu_rng)
        S_true = exact_solution_np(Xg, Tg, nu)  # [Nt,Nx]
        data.append({
            "nu": torch.tensor([[nu]], dtype=torch.float32),
            "S_true": torch.tensor(S_true, dtype=torch.float32),
        })
    return data

if LOAD_DATASETS_IF_EXIST and os.path.exists(dataset_path):
    with open(dataset_path, "rb") as f:
        payload = pickle.load(f)
    train_data = payload["train_data"]
    test_data  = payload["test_data"]
    print(f"[Data] Loaded dataset from {dataset_path}")
else:
    train_data = make_dataset(num_train, nu_range, Xg, Tg)
    test_data  = make_dataset(num_test,  nu_range, Xg, Tg)
    print("[Data] Generated new dataset.")
    if SAVE_DATASETS:
        with open(dataset_path, "wb") as f:
            pickle.dump({"train_data": train_data, "test_data": test_data}, f)
        print(f"[Data] Saved dataset to {dataset_path}")

# ============================================================
# 4) Models
# ============================================================
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
        return self.net(torch.cat([x, t, nu], dim=1))

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
        b  = self.branch(nu)
        tr = self.trunk(torch.cat([x, t], 1))
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
        return self.net(nu)

class BSNetLoss(nn.Module):
    def __init__(self, n_cp_t, n_cp_x, hidden=128):
        super().__init__()
        self.n_cp_t, self.n_cp_x = n_cp_t, n_cp_x
        self.ctrl = ControlPointNet(n_cp_t, n_cp_x, hidden)

    def forward_U(self, nu):
        vec = self.ctrl(nu)  # [B, n_cp_t*n_cp_x]
        return vec.view(-1, self.n_cp_t, self.n_cp_x)

# ============================================================
# 5) Precompute spline bases (depends on BSPLINE_CFG)
# ============================================================
n_cp_x = int(BSPLINE_CFG["n_cp_x"])
n_cp_t = int(BSPLINE_CFG["n_cp_t"])
d      = int(BSPLINE_CFG["degree"])

tk_x, Ln_x, Bx_np = BsKnots(n_cp_x, d, Nx)
Bx_d1_np, Bx_d2_np = BsKnots_derivatives(n_cp_x, d, Nx, Ln_x, tk_x)

tk_t, Ln_t, Bt_np = BsKnots(n_cp_t, d, Nt)
Bt_d1_np, Bt_d2_np = BsKnots_derivatives(n_cp_t, d, Nt, Ln_t, tk_t)

Bx     = torch.tensor(Bx_np,     dtype=torch.float32, device=device)
Bx_d1  = torch.tensor(Bx_d1_np,  dtype=torch.float32, device=device)
Bx_d2  = torch.tensor(Bx_d2_np,  dtype=torch.float32, device=device)
Bt     = torch.tensor(Bt_np,     dtype=torch.float32, device=device)
Bt_d1  = torch.tensor(Bt_d1_np,  dtype=torch.float32, device=device)
Bt_d2  = torch.tensor(Bt_d2_np,  dtype=torch.float32, device=device)

idx_x0 = 0
idx_x1 = Nx - 1
ic_true = torch.tensor(np.cos(np.pi * x_np), dtype=torch.float32, device=device)  # [Nx]

# ============================================================
# 6) Loss helpers
# ============================================================
def pinn_pde_residual(u, x, t, nu):
    u_t  = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x  = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - nu * u_xx

def relative_L2(pred, true, eps=1e-12):
    num = torch.sqrt(torch.mean((pred-true)**2))
    den = torch.sqrt(torch.mean(true**2)) + eps
    return (num/den).item()

# ============================================================
# 7) Build models + optimizers
# ============================================================
models = {
    "PINN": VanillaPINN(hidden=128).to(device),
    "PI-DeepONet": PIDeepONet(branch_hidden=128, trunk_hidden=128).to(device),
    "PI-BSNet": BSNetLoss(n_cp_t=n_cp_t, n_cp_x=n_cp_x, hidden=128).to(device),
}
optims = {k: optim.Adam(m.parameters(), lr=lr) for k, m in models.items()}

history = {k: {"total": [], "pde": [], "data": [], "ic": [], "bc": []} for k in models.keys()}
train_time = {}

# --------- Save global run config (for reproducibility) ----------
run_cfg = {
    "seed": SEED,
    "deterministic_cuda": DETERMINISTIC_CUDA,
    "device": str(device),
    "grid": {"Nx": Nx, "Nt": Nt, "x0": x0, "x1": x1, "T_max": T_max},
    "data": {"num_train": num_train, "num_test": num_test, "nu_range": list(nu_range), "dataset_path": dataset_path},
    "loss_weights": {"PDE": PDE_WEIGHT, "DATA": DATA_WEIGHT, "IC": IC_WEIGHT, "BC": BC_WEIGHT},
    "train": {"epochs": epochs, "lr": lr},
    "bspline": dict(BSPLINE_CFG),
    "train_models": dict(TRAIN_MODELS),
}
with open(os.path.join(save_dir, "run_config.json"), "w") as f:
    json.dump(run_cfg, f, indent=2)

# ============================================================
# 7.1) Load checkpoint helper
# ============================================================
def try_load_model(model_name: str, model: nn.Module, optimizer: optim.Optimizer):
    ckpt_file, opt_file, meta_file = ckpt_paths(model_name)
    if not os.path.exists(ckpt_file):
        return False, None

    sd = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(sd)

    if os.path.exists(opt_file):
        try:
            optimizer.load_state_dict(torch.load(opt_file, map_location=device))
        except Exception:
            pass

    meta = None
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = json.load(f)

    print(f"[Load] Loaded {model_name} from {ckpt_file}")
    if meta is not None and "bspline" in meta:
        print(f"[Load] {model_name} bspline meta: {meta['bspline']}")
    return True, meta

# ============================================================
# 7.2) Train or load each model
# ============================================================
for name, model in models.items():
    ckpt_file, opt_file, meta_file = ckpt_paths(name)

    do_train = bool(TRAIN_MODELS.get(name, True))
    if not do_train:
        loaded, meta = try_load_model(name, model, optims[name])
        if loaded:
            # Use recorded training time from meta (so tradeoff plot is meaningful)
            if meta is not None:
                train_time[name] = float(meta.get("train_time_sec", 0.0))
            else:
                train_time[name] = 0.0
            continue
        else:
            print(f"[Load] No checkpoint found for {name}. Will train.")
            do_train = True

    print(f"\n=== Training {name} ===")
    optimizer = optims[name]
    model.train()

    # Accurate wall time (and GPU sync if needed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for ep in tqdm(range(1, epochs+1), desc=name):
        optimizer.zero_grad()

        total_L = 0.0
        Lpde_acc = 0.0
        Ldata_acc = 0.0
        Lic_acc = 0.0
        Lbc_acc = 0.0

        for sample in train_data:
            nu = sample["nu"].to(device)         # [1,1]
            S_true = sample["S_true"].to(device) # [Nt,Nx]

            if name in ["PINN", "PI-DeepONet"]:
                xf = torch.tensor(Xg.reshape(-1,1), dtype=torch.float32, device=device, requires_grad=True)
                tf = torch.tensor(Tg.reshape(-1,1), dtype=torch.float32, device=device, requires_grad=True)
                nuf = nu.expand_as(xf).clone().detach().requires_grad_(True)

                u_pred = model(xf, tf, nuf)  # [Nt*Nx,1]
                u_true = S_true.reshape(-1,1)

                res = pinn_pde_residual(u_pred, xf, tf, nuf)
                Lpde = torch.mean(res**2)
                Ldata = torch.mean((u_pred - u_true)**2)

                x_ic = torch.tensor(x_np.reshape(-1,1), dtype=torch.float32, device=device, requires_grad=True)
                t_ic = torch.zeros_like(x_ic, device=device, requires_grad=True)
                nu_ic = nu.expand_as(x_ic).clone().detach().requires_grad_(True)
                u_ic_pred = model(x_ic, t_ic, nu_ic)
                u_ic_true = torch.cos(np.pi * x_ic)
                Lic = torch.mean((u_ic_pred - u_ic_true)**2)

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
                U = model.forward_U(nu)[0]  # [n_cp_t, n_cp_x]
                S_pred = bspline_eval(U, Bt, Bx)
                S_t, S_x, S_xx = bspline_derivs(U, Bt, Bx, Bt_d1, Bx_d1, Bt_d2, Bx_d2)

                Lpde = torch.mean((S_t - nu.item() * S_xx)**2)
                Ldata = torch.mean((S_pred - S_true)**2)
                Lic = torch.mean((S_pred[0, :] - ic_true)**2)
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

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    train_time[name] = elapsed

    # ---- Save checkpoint + optimizer + metadata ----
    torch.save(model.state_dict(), ckpt_file)
    torch.save(optimizer.state_dict(), opt_file)

    meta = {
        "model_name": name,
        "seed": SEED,
        "trained_this_run": True,
        "train_time_sec": float(elapsed),
        "epochs": epochs,
        "lr": lr,
        "loss_weights": {"PDE": PDE_WEIGHT, "DATA": DATA_WEIGHT, "IC": IC_WEIGHT, "BC": BC_WEIGHT},
        "ckpt_file": ckpt_file,
        "optim_file": opt_file,
    }
    if name == "PI-BSNet":
        meta["bspline"] = dict(BSPLINE_CFG)

    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Save] {name}: ckpt={ckpt_file}")
    print(f"[Save] {name}: meta={meta_file}")
    print(f"[Train] {name} finished in {elapsed:.1f}s")


# ============================================================
# 8) Testing + visualization (shared error range per nu + heat colormap)
# ============================================================
@torch.no_grad()
def eval_model_on_sample(model_name, model, nu, S_true):
    model.eval()
    if model_name == "PI-BSNet":
        # PI-BSNet predicts control points U, then evaluate B-spline surface
        U = model.forward_U(nu.to(device))[0]
        S_pred = bspline_eval(U, Bt, Bx)  # torch [Nt, Nx] or compatible
    else:
        xf = torch.tensor(Xg.reshape(-1, 1), dtype=torch.float32, device=device)
        tf = torch.tensor(Tg.reshape(-1, 1), dtype=torch.float32, device=device)
        nuf = nu.to(device).expand_as(xf)
        u_pred = model(xf, tf, nuf).reshape(Nt, Nx)
        S_pred = u_pred

    rel = relative_L2(S_pred, S_true.to(device))
    return S_pred.detach().cpu().numpy(), float(rel)

# --- visualize specific nu values ---
nu_list = [0.2, 0.4, 1.2]

single_case_errs = {}  # single_case_errs[nu][model_name] = relL2

for nu_val in nu_list:
    nu_plot = torch.tensor([[nu_val]], dtype=torch.float32)

    # Ground truth for this nu
    S_true_np = exact_solution_np(Xg, Tg, nu_val)   # shape [Nt, Nx]
    S_true_plot = torch.tensor(S_true_np, dtype=torch.float32)

    print(f"\n============================")
    print(f"Visualization for nu = {nu_val:.3f}")
    print(f"============================")

    single_case_errs[nu_val] = {}

    # ------------------------------------------------------------
    # Pass 1: run all models, cache predictions/errors, get shared error range
    # ------------------------------------------------------------
    cache = {}          # cache[name] = {"pred": pred_np, "err": err_np, "rel": rel}
    err_max = 0.0

    for name, model in models.items():
        pred_np, rel = eval_model_on_sample(name, model, nu_plot, S_true_plot)
        err_np = np.abs(pred_np - S_true_np)

        cache[name] = {"pred": pred_np, "err": err_np, "rel": rel}
        single_case_errs[nu_val][name] = rel

        # robust max (avoid nan issues)
        this_max = float(np.nanmax(err_np)) if np.isfinite(err_np).any() else 0.0
        err_max = max(err_max, this_max)

        print(f"[Test nu={nu_val:.3f}] {name}: rel-L2 = {rel:.3e}")

    # Shared error range for all methods at this nu
    err_vmin = 0.0
    err_vmax = err_max if err_max > 0 else 1e-12

    # ------------------------------------------------------------
    # Pass 2: plot each model using the SAME error color range
    # ------------------------------------------------------------
    for name in models.keys():
        pred_np = cache[name]["pred"]
        err_np  = cache[name]["err"]
        rel     = cache[name]["rel"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        # True
        im0 = axes[0].imshow(
            S_true_np, origin="lower", aspect="auto",
            extent=[x0, x1, 0, T_max]
        )
        axes[0].set_title("True")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
        plt.colorbar(im0, ax=axes[0])

        # Pred
        im1 = axes[1].imshow(
            pred_np, origin="lower", aspect="auto",
            extent=[x0, x1, 0, T_max]
        )
        axes[1].set_title(f"{name} Pred")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("t")
        plt.colorbar(im1, ax=axes[1])

        # Abs Error (HEAT colormap + shared vmin/vmax for this nu)
        im2 = axes[2].imshow(
            err_np, origin="lower", aspect="auto",
            extent=[x0, x1, 0, T_max],
            cmap="hot", vmin=err_vmin, vmax=err_vmax
        )
        axes[2].set_title(f"Abs Error (rel-L2={rel:.2e})")
        axes[2].set_xlabel("x"); axes[2].set_ylabel("t")
        plt.colorbar(im2, ax=axes[2])

        fig.suptitle(
            f"nu={nu_val:.3f}  |  shared error range: [{err_vmin:.2e}, {err_vmax:.2e}]"
        )

        # include nu in filename so plots don't overwrite each other
        fig_path = os.path.join(save_dir, f"heatmap_nu{nu_val:.3f}_{model_tag(name)}.png")
        fig.savefig(fig_path, dpi=200)

        plt.show()
        plt.close(fig)

# (optional) print a compact summary table for these nus
print("\n=== Single-case rel-L2 summary (requested nus) ===")
for nu_val in nu_list:
    row = " | ".join([f"{name}: {single_case_errs[nu_val][name]:.3e}" for name in models.keys()])
    print(f"nu={nu_val:.3f} -> {row}")

avg_err = {}
for name, model in models.items():
    rels = []
    for sample in test_data:
        _, rel = eval_model_on_sample(name, model, sample["nu"], sample["S_true"])
        rels.append(rel)
    avg_err[name] = float(np.mean(rels))
    print(f"[Avg Test] {name}: rel-L2 = {avg_err[name]:.3e}")

# If you want a single-case dict for summary, use the nu_list results:
# (e.g., store per-nu errors instead of a single scalar)
errs = {}  # keep key name for backward compatibility in your summary format
for name in models.keys():
    errs[name] = float(single_case_errs[nu_list[0]][name])  # pick nu_list[0] = 0.2 as "single-case"

# ============================================================
# 9) Save run summary (times + errors + cfg)
# ============================================================
summary = {
    "seed": SEED,
    "dataset_path": dataset_path,
    "device": str(device),
    "epochs": epochs,
    "lr": lr,
    "loss_weights": {"PDE": PDE_WEIGHT, "DATA": DATA_WEIGHT, "IC": IC_WEIGHT, "BC": BC_WEIGHT},
    "bspline_cfg": dict(BSPLINE_CFG),
    "train_time_sec": {k: float(train_time.get(k, 0.0)) for k in models.keys()},
    "avg_test_relL2": {k: float(avg_err.get(k, np.nan)) for k in models.keys()},
    "single_case_relL2": {k: float(errs.get(k, np.nan)) for k in models.keys()},
    "ckpt_files": {k: ckpt_paths(k)[0] for k in models.keys()},
    "trained_this_run": dict(TRAIN_MODELS),
}
summary_path = os.path.join(save_dir, f"summary_{model_tag('PI-BSNet')}.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print("[Save] Summary:", summary_path)

print("Done. Plots saved to:", save_dir)
