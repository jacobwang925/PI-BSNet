import logging
from typing import Dict

import opt_einsum as oe
import torch

from bsnet.utils import BsKnots, BsKnots_derivatives

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Spline:
    """Stores information related to the spline and returns splines for given C.P.s"""

    def __init__(
        self,
        n_ctrl_pts_time,
        n_ctrl_pts_state,
        dimension,
        order,
        n_points,
        min_t,
        max_t,
        min_x,
        max_x,
    ):

        print(n_ctrl_pts_state, max_t, max_x, min_t, min_x)
        self.n_ctrl_pts_time = n_ctrl_pts_time
        self.n_ctrl_pts_state = n_ctrl_pts_state
        self.dimension = dimension
        self.order = order
        self.n_points = n_points
        self.min_t = min_t
        self.max_t = max_t
        self.min_x = min_x
        self.max_x = max_x

        """initialize bspline matrices"""
        self.t = torch.linspace(self.min_t, self.max_t, self.n_points)
        self.x = torch.linspace(self.min_x, self.max_x, self.n_points)
        self.tk_t, self.Ln_t, self.bit_t = BsKnots(
            self.n_ctrl_pts_time, self.order, self.n_points
        )
        self.bit_t_d, self.bit_t_dd = BsKnots_derivatives(
            self.n_ctrl_pts_time, self.order, len(self.t), self.Ln_t, self.tk_t
        )

        tk_x, Ln_x, bit_x = BsKnots(self.n_ctrl_pts_state, self.order, self.n_points)
        bit_x_d, bit_x_dd = BsKnots_derivatives(
            self.n_ctrl_pts_state, self.order, len(self.x), Ln_x, tk_x
        )

        for i in "xyz"[: self.dimension]:
            setattr(self, i, self.x)
            setattr(self, f"tk_{i}", tk_x)
            setattr(self, f"Ln_{i}", Ln_x)
            setattr(self, f"bit_{i}", bit_x)
            setattr(self, f"bit_{i}_d", bit_x_d)
            setattr(self, f"bit_{i}_dd", bit_x_dd)

    def make_surface(self, inner_matrix):
        # Generate the B-spline surface with the predicted control points
        if inner_matrix.ndim == 5:
            y = torch.einsum(
                "qijkl,ti,xj,yk,zl->qtxyz",
                inner_matrix,
                self.bit_t,
                self.bit_x,
                self.bit_y,
                self.bit_z,
            )
        if inner_matrix.ndim == 4:
            y = torch.einsum(
                "qjkl,xj,yk,zl->qxyz", inner_matrix, self.bit_x, self.bit_y, self.bit_z
            )
        return y

    def computebsderivatives(self, U_full):
        path_info = oe.contract_path(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.bit_t,
            self.bit_x,
            self.bit_y,
            self.bit_z,
        )[0]
        B_surface = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.bit_t,
            self.bit_x,
            self.bit_y,
            self.bit_z,
            optimize=path_info,
        )
        B_surface_t = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.bit_t_d,
            self.bit_x,
            self.bit_y,
            self.bit_z,
            optimize=path_info,
        )
        B_surface_xx = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.bit_t,
            self.bit_x_dd,
            self.bit_y,
            self.bit_z,
            optimize=path_info,
        )
        B_surface_yy = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.bit_t,
            self.bit_x,
            self.bit_y_dd,
            self.bit_z,
            optimize=path_info,
        )
        B_surface_zz = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.bit_t,
            self.bit_x,
            self.bit_y,
            self.bit_z_dd,
            optimize=path_info,
        )
        # Laplacian (sum of second derivatives in each spatial direction)
        B_surface_laplacian = B_surface_xx + B_surface_yy + B_surface_zz
        return B_surface, B_surface_t, B_surface_laplacian
