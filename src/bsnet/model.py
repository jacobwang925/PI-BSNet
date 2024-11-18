import logging
import os
from abc import abstractmethod
from datetime import datetime
from typing import Dict

import lightning as L
import torch
import torch.optim as optim

from bsnet.pde import PDE
from bsnet.spline import Spline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class BSNet(L.LightningModule):
    def __init__(
        self,
        pde: PDE,
        bspline: Spline,
        model: torch.nn.Module,
        dimension: int = 1,
    ):
        super().__init__()
        self.bspline = bspline
        self.dimension = dimension
        # self.model_params = kwargs  # Store any additional parameters

        self.pde = pde
        self.model = model
        self.model_name = model.name
        self.run_version = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.test_pred_full_solution = []
        self.test_pred_ic0_solution = []
        self.test_true_input_params = []
        self.test_true_ic0_solution = []

    def forward(self, x):
        full, at_ic0 = self.model(x)

        # hardcoded bcs (see also in data)
        full[:, :, :, :, [0, -1]] = 1
        at_ic0[:, :, :, [0, -1]] = 1

        return full, at_ic0

    def training_step(self, batch, batch_idx):
        input_params = batch["X"]
        output_at_time_0 = batch["y"]
        full_pde, pde_at_time_0 = self(input_params)
        self.training_loss = self.loss(full_pde, pde_at_time_0, output_at_time_0)
        self.log("train_loss_epoch", self.training_loss, prog_bar=True)
        return self.training_loss

    def validation_step(self, batch, batch_idx):
        input_params = batch["X"]
        output_at_time_0 = batch["y"]
        full_pde, pde_at_time_0 = self(input_params)
        self.validation_loss = self.loss(full_pde, pde_at_time_0, output_at_time_0)
        self.log("valid_loss_epoch", self.validation_loss, prog_bar=True)
        return self.validation_loss

    def test_step(self, batch, batch_idx):
        input_params = batch["X"]
        output_at_time_0 = batch["y"]
        full_pde, pde_at_time_0 = self(input_params)
        self.test_loss = self.loss(full_pde, pde_at_time_0, output_at_time_0)
        self.test_pred_full_solution.append(full_pde)
        self.test_pred_ic0_solution.append(pde_at_time_0)
        self.test_true_input_params.append(input_params)
        self.test_true_ic0_solution.append(output_at_time_0)

    def on_train_epoch_end(self):
        """run at the end of each training run"""
        # log epoch metric
        self.log(
            "train_loss_epoch",
            self.training_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "valid_loss_epoch",
            self.validation_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def on_test_epoch_end(self):
        """what to do when testing"""

        test_pred_full_solution = torch.concat(self.test_pred_full_solution)
        test_pred_ic0_solution = torch.concat(self.test_pred_ic0_solution)
        test_true_input_params = torch.concat(self.test_true_input_params)
        test_true_ic0_solution = torch.concat(self.test_true_ic0_solution)

        # Extract directory and filename from checkpoint path
        checkpoint_path = self.trainer.ckpt_path
        checkpoint_dir = os.path.dirname(checkpoint_path).replace(
            "logs/lightning_logs/version_0/checkpoints", "test_results"
        )
        print(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_name = os.path.basename(checkpoint_path).replace(".ckpt", "")

        # Create a dynamic filename based on the checkpoint
        filename = os.path.join(
            checkpoint_dir, f"{checkpoint_name}_test_results_{self.run_version}.pt"
        )
        predictions = {
            "pred_full_solution": test_pred_full_solution,
            "pred_ic0_solution": test_pred_ic0_solution,
            "true_input_params": test_true_input_params,
            "true_ic0_solution": test_true_ic0_solution,
        }
        spline = self.bspline.get_init_params()
        pde = self.pde.get_init_params()

        save_all = predictions | spline | pde  # note! need python > 3.9
        # Save the results
        torch.save(save_all, filename)

    def loss(self, y_hat, y_ic, ic):

        loss_on_ic_ctrl = torch.mean(torch.sum((y_ic - ic) ** 2, axis=(1, 2, 3)))

        # if loss_on_ic_ctrl > 100:
        #    return loss_on_ic_ctrl

        ic_surface = self.bspline.make_surface(ic)
        y_ic_surface = self.bspline.make_surface(y_ic)
        ic_loss = torch.mean(
            torch.sum((y_ic_surface - ic_surface) ** 2, axis=(1, 2, 3))
        )

        # if ic_loss > 50:
        #    return loss_on_ic_ctrl + ic_loss

        u_full = y_hat
        B_surface, B_s_t, B_s_xx = self.bspline.computebsderivatives(u_full)
        pde_loss = torch.mean(
            torch.sum(
                (self.pde.pde_residual(u=None, dudt=B_s_t, dudx=None, dudxdx=B_s_xx))
                ** 2,
                axis=(1, 2, 3),
            )
        )

        neumann_loss_xy = torch.mean(
            torch.sum((B_s_t[:, :, [0, -1], :, :]) ** 2, axis=(2, 3, 4))
        )
        neumann_loss_xy += torch.mean(
            torch.sum((B_s_t[:, :, :, [0, -1], :]) ** 2, axis=(2, 3, 4))
        )

        return loss_on_ic_ctrl + ic_loss + neumann_loss_xy + 10 * pde_loss
