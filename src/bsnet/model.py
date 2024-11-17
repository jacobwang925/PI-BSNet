import logging
from abc import abstractmethod
from datetime import datetime
from typing import Dict

import lightning as L
import torch
import torch.optim as optim

from bsnet.pde import PDE
from bsnet.spline import Spline

logging.basicConfig(level=logging.DEBUG)
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

    def forward(self, x):
        return self.model(x)

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

    def on_train_epoch_end(self):
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

    def loss(self, y_hat, y_ic, ic):
        y_hat[:, :, :, :, [0, -1]] = 1
        y_ic[:, :, :, [0, -1]] = 1
        ic[:, :, :, [0, -1]] = 1

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

        U_full = y_hat
        B_surface, B_s_t, B_s_xx = self.bspline.computebsderivatives(U_full)
        pde_loss = torch.mean(
            torch.sum(
                (self.pde.pde_residual(u=0, dudt=B_s_t, dudx=0, dudxdx=B_s_xx)) ** 2,
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
