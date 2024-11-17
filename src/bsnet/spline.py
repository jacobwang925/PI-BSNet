import logging
from typing import Dict
import opt_einsum as oe
import torch

from bsnet.utils import BsKnots, BsKnots_derivatives

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



class Spline:
    """ Stores information related to the spline and returns splines for given C.P.s
    """
    def __init__(
        self,
        n_ctrl_pts_time,
        n_ctrl_pts_state,
        dimension,
        order,
        n_points,
        min_T,
        max_T,
        min_X,
        max_X,
    ):

        print(n_ctrl_pts_state)
        self.n_ctrl_pts_time = n_ctrl_pts_time
        self.n_ctrl_pts_state = n_ctrl_pts_state
        self.dimension = dimension
        self.order = order
        self.n_points = n_points
        self.min_T = min_T
        self.max_T = max_T
        self.min_X = min_X
        self.max_X = max_X

        self.initialize_bspline_matrices()

    def initialize_bspline_matrices(self):
        """initialize bspline matrices"""
        self.t = torch.linspace(self.min_T, self.max_T, self.n_points)
        self.x = torch.linspace(self.min_X, self.max_X, self.n_points)
        self.tk_t, self.Ln_t, self.Bit_t = BsKnots(
            self.n_ctrl_pts_time, self.order, self.n_points
        )
        self.Bit_t_d, self.Bit_t_dd = BsKnots_derivatives(
            self.n_ctrl_pts_time, self.order, len(self.t), self.Ln_t, self.tk_t
        )

        tk_x, Ln_x, Bit_x = BsKnots(self.n_ctrl_pts_state, self.order, self.n_points)
        Bit_x_d, Bit_x_dd = BsKnots_derivatives(
            self.n_ctrl_pts_state, self.order, len(self.x), Ln_x, tk_x
        )

        for i in "xyz"[: self.dimension]:
            setattr(self, i, self.x)
            setattr(self, f"tk_{i}", tk_x)
            setattr(self, f"Ln_{i}", Ln_x)
            setattr(self, f"Bit_{i}", Bit_x)
            setattr(self, f"Bit_{i}_d", Bit_x_d)
            setattr(self, f"Bit_{i}_dd", Bit_x_dd)

    def make_surface(self, inner_matrix):
        # Generate the B-spline surface with the predicted control points
        if inner_matrix.ndim == 5:
            y = torch.einsum(
                "qijkl,ti,xj,yk,zl->qtxyz",
                inner_matrix,
                self.Bit_t,
                self.Bit_x,
                self.Bit_y,
                self.Bit_z,
            )
        if inner_matrix.ndim == 4:
            y = torch.einsum(
                "qjkl,xj,yk,zl->qxyz", inner_matrix, self.Bit_x, self.Bit_y, self.Bit_z
            )
        return y

    def computebsderivatives(self, U_full):
        path_info = oe.contract_path(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.Bit_t,
            self.Bit_x,
            self.Bit_y,
            self.Bit_z,
        )[0]
        B_surface = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.Bit_t,
            self.Bit_x,
            self.Bit_y,
            self.Bit_z,
            optimize=path_info,
        )
        B_surface_t = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.Bit_t_d,
            self.Bit_x,
            self.Bit_y,
            self.Bit_z,
            optimize=path_info,
        )
        B_surface_xx = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.Bit_t,
            self.Bit_x_dd,
            self.Bit_y,
            self.Bit_z,
            optimize=path_info,
        )
        B_surface_yy = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.Bit_t,
            self.Bit_x,
            self.Bit_y_dd,
            self.Bit_z,
            optimize=path_info,
        )
        B_surface_zz = oe.contract(
            "qijkl,ti,xj,yk,zl->qtxyz",
            U_full,
            self.Bit_t,
            self.Bit_x,
            self.Bit_y,
            self.Bit_z_dd,
            optimize=path_info,
        )
        # Laplacian (sum of second derivatives in each spatial direction)
        B_surface_laplacian = B_surface_xx + B_surface_yy + B_surface_zz
        return B_surface, B_surface_t, B_surface_laplacian




