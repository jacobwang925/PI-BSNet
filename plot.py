import fire
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bsnet.spline import Spline
from bsnet.pde import PDE
import logging

# mpl_logger = logging.getLogger(*matplotlib*)
# mpl_logger.setLevel(logging.WARNING)


def get_results(results_file):
    results = torch.load(results_file)
    spline_keys = [
        "n_ctrl_pts_time",
        "n_ctrl_pts_state",
        "dimension",
        "order",
        "n_points",
        "min_t",
        "max_t",
        "min_x",
        "max_x",
    ]
    pde_keys = ["D"]
    spline_params = dict((k, results[k]) for k in spline_keys if k in results)
    pde_params = dict((k, results[k]) for k in pde_keys if k in results)
    bspline = Spline(**spline_params)
    pde = PDE(**pde_params)
    pred_full_solution = results["pred_full_solution"]
    pred_ic0_solution = results["pred_ic0_solution"]
    true_input_params = results["true_input_params"]
    true_ic0_solution = results["true_ic0_solution"]
    print(pred_ic0_solution.shape, pred_full_solution.shape)
    pred_ic0_surface = bspline.make_surface(pred_ic0_solution)
    true_ic0_surface = bspline.make_surface(true_ic0_solution)

    return (
        bspline,
        pred_full_solution,
        pred_ic0_solution,
        true_input_params,
        true_ic0_solution,
        pred_ic0_surface,
        true_ic0_surface,
        pde,
    )


def run(results_file_train: Path, results_file_test: Path, pick=5):

    (
        train_bs,
        pred_full_sol_train,
        pred_ic0_sol_train,
        true_full_sol_train,
        true_ic0_sol_train,
        pred_ic0_surf_train,
        true_ic0_surf_train,
        train_pde,
    ) = get_results(results_file_train)

    (
        test_bs,
        pred_full_sol_test,
        pred_ic0_sol_test,
        true_full_sol_test,
        true_ic0_sol_test,
        pred_ic0_surf_test,
        true_ic0_surf_test,
        test_pde,
    ) = get_results(results_file_train)

    plt.gca().xaxis.tick_bottom()

    # 1
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))

    diff_ic_train = (
        ((true_ic0_sol_train - pred_ic0_sol_train) / (true_ic0_sol_train + 0.000001))
        .detach()
        .numpy()
    )
    mean_diff_ic_train = np.mean(diff_ic_train**2, axis=(1, 2, 3))
    std_diff_ic_train = np.std(diff_ic_train**2, axis=(1, 2, 3))
    ax[0, 0].scatter(
        np.linspace(1, len(mean_diff_ic_train), len(mean_diff_ic_train)),
        mean_diff_ic_train,
        label="Train",
    )

    diff_ic_test = (
        (true_ic0_sol_test - pred_ic0_sol_test / (true_ic0_sol_test + 0.000001))
        .detach()
        .numpy()
    )
    mean_diff_ic_test = np.mean(diff_ic_test**2, axis=(1, 2, 3))
    std_diff_ic_test = np.std(diff_ic_test**2, axis=(1, 2, 3))
    ax[0, 0].scatter(
        np.linspace(1, len(mean_diff_ic_test), len(mean_diff_ic_test)),
        mean_diff_ic_test,
        label="Test",
    )

    ax[0, 0].set_title("<$(I.C._{pred} - I.C._{true})^2$>")

    # ---
    ax01 = ax[0, 1].matshow(
        (true_ic0_sol_test - pred_ic0_sol_test).detach()[
            pick, :, :, test_bs.n_points // 2
        ]
    )
    plt.colorbar(ax01)
    ax[0, 1].set_title(
        " $1^{st}$ test point $(I.C._{pred} - I.C._{true})^2$ (XY plane)"
    )

    ax02 = ax[1, 0].matshow(
        (pred_ic0_sol_test).detach()[pick, :, :, test_bs.n_points // 2].T
    )
    plt.colorbar(ax02)

    ax[1, 0].set_title(" $1^{st}$ test point $I.C._{pred}$ (XY plane)")
    ax[1, 0].set_xlabel("X (cm)")
    ax[1, 0].set_ylabel("Y (cm)")
    ax[1, 0].xaxis.set_ticks_position("bottom")
    ax[1, 0].tick_params(top=False, bottom=True)

    ax02 = ax[1, 1].matshow(
        (pred_ic0_sol_test).detach()[pick, :, test_bs.n_points // 2, :].T
    )
    plt.colorbar(ax02)
    ax[1, 1].set_title(" $1^{st}$ test point $I.C._{pred}$ (XZ plane)")
    ax[1, 1].set_xlabel("X (cm)")
    ax[1, 1].set_ylabel("Z (cm)")
    ax[1, 1].xaxis.set_ticks_position("bottom")
    ax[1, 1].tick_params(top=False, bottom=True)
    fig.savefig("figure_1_icbc.pdf")

    # 2
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    print(pred_full_sol_train[pick, :].shape)
    (pred_surf_train, pred_dt_surf_train, pred_laplace_surf_train) = (
        train_bs.computebsderivatives(pred_full_sol_train)
    )
    (pred_surf_test, pred_dt_surf_test, pred_laplace_surf_test) = (
        train_bs.computebsderivatives(pred_full_sol_test)
    )

    surf_heat_diff_pred_train = train_pde.pde_residual(
        u=pred_surf_train,
        dudt=pred_dt_surf_train,
        dudx=None,
        dudxdx=pred_laplace_surf_train,
    ).detach()
    mean_surf_heat_diff_pred_train = torch.mean(
        surf_heat_diff_pred_train**2, axis=(1, 2, 3, 4)
    )
    # std_surf_heat_diff_pred_train = torch.std(surf_heat_diff_pred_train**2, axis = (1,2,3,4))
    Q = len(mean_surf_heat_diff_pred_train)
    ax[1, 0].scatter(
        [1] * Q + np.random.rand(Q) / 3, mean_surf_heat_diff_pred_train, label="Train"
    )
    # ax[1,0].errorbar(np.linspace(1,Q,Q), mean_surf_heat_diff_pred_train, std_surf_heat_diff_pred_train)
    ax[1, 0].set_ylim(-0.001, 0.001)

    surf_heat_diff_pred_test = test_pde.pde_residual(
        u=pred_surf_test,
        dudt=pred_dt_surf_test,
        dudx=None,
        dudxdx=pred_laplace_surf_test,
    ).detach()
    mean_surf_heat_diff_pred_test = torch.mean(
        surf_heat_diff_pred_test**2, axis=(1, 2, 3, 4)
    )
    # std_surf_heat_diff_pred_test = torch.std(surf_heat_diff_pred_test**2, axis = (1,2,3,4))
    Q = len(mean_surf_heat_diff_pred_test)
    ax[1, 0].scatter(
        [2] * Q + np.random.rand(Q) / 3, mean_surf_heat_diff_pred_test, label="Test"
    )
    # ax[1,0].errorbar(np.linspace(1,Q,Q), mean_surf_heat_diff_pred_test, std_surf_heat_diff_pred_test)
    ax[1, 0].xaxis.set_ticks_position("bottom")
    ax[1, 0].tick_params(top=False, bottom=True)
    ax[1, 0].set_title("mean squared residual over all training and test inputs")
    ax[1, 0].set_ylim(0.00001, 0.0004)
    ax[1, 0].set_xlim(0.5, 2.5)
    ax[1, 0].set_xticks([1, 2])
    ax[1, 0].set_xticklabels(["Training Data", "Test Data"])

    # Adjust the position of ax[1, 0] to be half the width
    pos1 = ax[1, 0].get_position()  # Get the original position
    new_pos1 = [pos1.x0, pos1.y0, pos1.width / 1.5, pos1.height]  # Modify the width
    ax[1, 0].set_position(new_pos1)  # Set the new position

    aa = ax[0, 0].matshow(
        surf_heat_diff_pred_test[pick, -1, :, :, test_bs.n_points // 2].T
    )
    plt.colorbar(aa)
    ax[0, 0].set_title(
        r"$U(t=1\,s,x,y,z=0.5\,cm),\;\partial_t U - D\cdot \Delta U$ X-Y slice"
    )
    ax[0, 0].set_ylabel("Y (cm)")
    ax[0, 0].set_xlabel("X (cm)")
    ax[0, 0].set_xticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[0, 0].set_yticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[0, 0].set_xticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[0, 0].set_yticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[0, 0].xaxis.set_ticks_position("bottom")
    ax[0, 0].tick_params(top=False, bottom=True)

    ab = ax[0, 1].matshow(
        surf_heat_diff_pred_test[
            pick, :, :, test_bs.n_points // 2, test_bs.n_points // 2
        ]
    )
    ax[0, 1].set_title(
        r"$U(t,x\,,y=0.5\,cm,z=0.5\,cm),\;\partial_t U - D\cdot \Delta U$  T-X slice"
    )
    ax[0, 1].set_xlabel("X (cm)")
    ax[0, 1].set_ylabel("Time (s)")
    ax[0, 1].set_xticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[0, 1].set_yticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[0, 1].set_xticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[0, 1].set_yticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[0, 1].xaxis.set_ticks_position("bottom")
    ax[0, 1].tick_params(top=False, bottom=True)
    plt.colorbar(ab)

    print(torch.sum(pred_full_sol_test[:, :, :, :, [0, -1]]).detach())
    pred_full_time_test = test_bs.make_surface(pred_full_sol_test).detach().numpy()

    print(np.sum(pred_full_time_test[:, :, :, :, [0, -1]] ** 2))
    fig.savefig("figure_2_heat_eq.pdf")
    # 3
    fig, ax = plt.subplots(1, 3, figsize=(21, 4))
    a = ax[0].matshow(
        pred_full_time_test[0, :, :, test_bs.n_points // 3, test_bs.n_points // 3]
    )
    ax[0].set_title("<U(t,x,y,z)>$_{xz}$ T-X profile")
    ax[0].set_xlabel("X (cm)")
    ax[0].set_ylabel("T")
    ax[0].set_xticks([i * 3 for i in range(test_bs.n_points // 3 + 1)])
    ax[0].set_yticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[0].set_xticklabels(
        [f"{i*3/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 3 + 1)]
    )
    ax[0].set_yticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[0].xaxis.set_ticks_position("bottom")
    ax[0].tick_params(top=False, bottom=True)
    plt.colorbar(a)

    b = ax[1].matshow(np.mean(pred_full_time_test[pick], axis=(1, 3)))
    ax[1].set_title("<U(t,x,y,z)>$_{xz}$ T-Y profile")
    ax[1].set_xlabel("Y (cm)")
    ax[1].set_ylabel("T")
    ax[1].set_xticks([i * 3 for i in range(test_bs.n_points // 3 + 1)])
    ax[1].set_yticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[1].set_xticklabels(
        [f"{i*3/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 3 + 1)]
    )
    ax[1].set_yticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[1].xaxis.set_ticks_position("bottom")
    ax[1].tick_params(top=False, bottom=True)
    plt.colorbar(b)

    c = ax[2].matshow(np.mean(pred_full_time_test[pick], axis=(1, 2)))
    ax[2].set_title("<U(t,x,y,z)>$_{xy}$ T-Z profile")
    ax[2].set_xlabel("Z (cm)")
    ax[2].set_ylabel("T")
    ax[2].set_xticks([i * 3 for i in range(test_bs.n_points // 3 + 1)])
    ax[2].set_yticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[2].set_xticklabels(
        [f"{i*3/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 3 + 1)]
    )
    ax[2].set_yticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )

    ax[2].xaxis.set_ticks_position("bottom")
    ax[2].tick_params(top=False, bottom=True)
    plt.colorbar(c)
    fig.savefig("figure_3_profiles.pdf")

    # 4
    # ax[1,1].colorbar()
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    g = ax[0].matshow(pred_full_time_test[pick, 1, :, test_bs.n_points // 2, :])
    h = ax[1].matshow(pred_full_time_test[pick, 1, :, :, test_bs.n_points // 2])
    ax[0].set_title(r"U(t=0.05s,x,y=$\frac{1}{2}$,z) X-Z slice")
    ax[1].set_title(r"U(t=0.05s,x,y,z=$\frac{1}{2}$) X-Y slice")

    ax[0].set_xlabel("X (cm)")
    ax[0].set_ylabel("Z (cm)")
    ax[1].set_xlabel("X (cm)")
    ax[1].set_ylabel("Y (cm)")

    ax[0].set_xticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[0].set_yticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[1].set_xticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[1].set_yticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[0].set_xticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[0].set_yticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[0].xaxis.set_ticks_position("bottom")
    ax[0].tick_params(top=False, bottom=True)
    ax[1].set_xticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[1].set_yticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[1].xaxis.set_ticks_position("bottom")
    ax[1].tick_params(top=False, bottom=True)
    ax[1].xaxis.tick_bottom()
    plt.colorbar(g)
    plt.colorbar(h)

    fig.savefig("figure_4_slices_t005.pdf")

    # 5
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    k = ax[0].matshow(pred_full_time_test[pick, -1, :, test_bs.n_points // 2, :])
    l = ax[1].matshow(pred_full_time_test[pick, -1, :, :, test_bs.n_points // 2])
    ax[0].set_title(r"U(t=1 s,x,y=$\frac{1}{2}$,z) X-Z slice")
    ax[0].set_xlabel("X (cm)")
    ax[0].set_ylabel("Z (cm)")
    ax[0].set_xticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[0].set_yticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[0].set_xticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[0].set_yticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[0].xaxis.set_ticks_position("bottom")
    ax[0].tick_params(top=False, bottom=True)

    ax[1].set_title(r"U(t=1 s,x,y,z=$\frac{1}{2}$) X-Y slice")
    ax[1].set_xlabel("X (cm)")
    ax[1].set_ylabel("Y (cm)")
    ax[1].set_xticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[1].set_yticks([i * 2 for i in range(test_bs.n_points // 2 + 1)])
    ax[1].set_xticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[1].set_yticklabels(
        [f"{i*2/test_bs.n_points:0.2f}" for i in range(test_bs.n_points // 2 + 1)]
    )
    ax[1].xaxis.set_ticks_position("bottom")
    ax[1].tick_params(top=False, bottom=True)
    plt.colorbar(k)
    plt.colorbar(l)
    fig.savefig("figure_5_slices_t095.pdf")

    plt.show()


if __name__ == "__main__":
    fire.Fire(run)
