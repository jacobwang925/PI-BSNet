import logging
from typing import Dict

import opt_einsum as oe
import torch

from bsnet.utils import BsKnots, BsKnots_derivatives

logging.basicConfig(level=logging.WARNING)
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
        self.tk_t, self.ln_t, self.bit_t = BsKnots(
            self.n_ctrl_pts_time, self.order, self.n_points
        )
        self.bit_t_d, self.bit_t_dd = BsKnots_derivatives(
            self.n_ctrl_pts_time, self.order, len(self.t), self.ln_t, self.tk_t
        )

        tk_x, ln_x, bit_x = BsKnots(self.n_ctrl_pts_state, self.order, self.n_points)
        bit_x_d, bit_x_dd = BsKnots_derivatives(
            self.n_ctrl_pts_state, self.order, len(self.x), ln_x, tk_x
        )
        self.tk_x = tk_x
        self.ln_x = ln_x
        self.bit_x = bit_x
        self.bit_x_d = bit_x_d
        self.bit_x_dd = bit_x_dd

        if self.dimension > 1:
            self.tk_y = tk_x
            self.ln_y = ln_x
            self.bit_y = bit_x
            self.bit_y_d = bit_x_d
            self.bit_y_dd = bit_x_dd

            if self.dimension > 2:
                self.tk_z = tk_x
                self.ln_z = ln_x
                self.bit_z = bit_x
                self.bit_z_d = bit_x_d
                self.bit_z_dd = bit_x_dd

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

    def computebsderivatives(self, u_full):
        path_info = oe.contract_path(
            "qijkl,ti,xj,yk,zl->qtxyz",
            u_full,
            self.bit_t,
            self.bit_x,
            self.bit_y,
            self.bit_z,
        )[0]
        b_surface = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            u_full,
            self.bit_t,
            self.bit_x,
            self.bit_y,
            self.bit_z,
            optimize=path_info,
        )
        b_surface_t = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            u_full,
            self.bit_t_d,
            self.bit_x,
            self.bit_y,
            self.bit_z,
            optimize=path_info,
        )
        b_surface_xx = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            u_full,
            self.bit_t,
            self.bit_x_dd,
            self.bit_y,
            self.bit_z,
            optimize=path_info,
        )
        b_surface_yy = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            u_full,
            self.bit_t,
            self.bit_x,
            self.bit_y_dd,
            self.bit_z,
            optimize=path_info,
        )
        b_surface_zz = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            u_full,
            self.bit_t,
            self.bit_x,
            self.bit_y,
            self.bit_z_dd,
            optimize=path_info,
        )
        # Laplacian (sum of second derivatives in each spatial direction)
        b_surface_laplacian = b_surface_xx + b_surface_yy + b_surface_zz
        return b_surface, b_surface_t, b_surface_laplacian

    def get_init_params(self):
        return vars(self)
