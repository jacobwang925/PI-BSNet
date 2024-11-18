# pylint: disable=invalid-name
from typing import List
import numpy as np
import torch
from scipy.integrate import quad


# B-spline basis function
def BsFun(i: int, d: int, t: float, Ln: List[float]) -> float:
    """recursive function

    Args:
        i (int): _description_
        d (int): _description_
        t (float): _description_
        Ln (List[float]): _description_

    Returns:
        _type_: _description_
    """
    if d == 0:
        return 1.0 if Ln[i - 1] <= t < Ln[i] else 0.0
    else:
        a = (
            0
            if (Ln[d + i - 1] - Ln[i - 1]) == 0
            else (t - Ln[i - 1]) / (Ln[d + i - 1] - Ln[i - 1])
        )
        b = 0 if (Ln[d + i] - Ln[i]) == 0 else (Ln[d + i] - t) / (Ln[d + i] - Ln[i])
        return a * BsFun(i, d - 1, t, Ln) + b * BsFun(i + 1, d - 1, t, Ln)


# B-spline knots and basis matrix
def BsKnots(n_cp: int, d: int, Ns: int):
    """B-spline knots and basis matrix

    Args:
        n_cp (int): number of control points
        d (int): _description_
        Ns (int): number of points

    Returns:
        _type_: _description_
    """
    n_knots = n_cp + d + 1
    Ln = torch.zeros(n_knots)

    # Construct the knots vector
    for i in range(d + 1, n_knots - d - 1):
        Ln[i] = i - d
    Ln[n_knots - d - 1 :] = n_cp - d  # The last d+1 elements should be the same

    # Parameter vector (linearly spaced)
    tk = torch.zeros(Ns)
    for i in range(1, Ns):
        tk[i] = tk[i - 1] + Ln[-1] / (Ns - 1)

    # B-spline basis matrix
    Bit = torch.zeros((Ns, n_cp))
    for j in range(n_cp):
        for i in range(Ns):
            Bit[i, j] = BsFun(j + 1, d, tk[i], Ln)

    Bit[Ns - 1, n_cp - 1] = 1

    return tk, Ln, Bit


# B-spline derivative basis function (first derivative)
def BsKnots_derivatives(n_cp, d, Ns, Ln, tk):
    # First derivative
    bit_derivative = torch.zeros((Ns, n_cp))
    for j in range(n_cp):
        for i in range(Ns):
            bit_derivative[i, j] = BsFun_derivative(j + 1, d, tk[i], Ln)

    # Second derivative
    bit_second_derivative = torch.zeros((Ns, n_cp))
    for j in range(n_cp):
        for i in range(Ns):
            bit_second_derivative[i, j] = BsFun_second_derivative(j + 1, d, tk[i], Ln)

    return bit_derivative, bit_second_derivative


def BsFun_derivative(i: int, d: int, t, Ln: List[float]):
    """get bspline function derivatives

    Args:
        i (int): _description_
        d (int): _description_
        t (_type_): _description_
        Ln (List[float]): _description_

    Returns:
        _type_: _description_
    """
    if d == 0:
        return 0.0

    a = 0 if (Ln[d + i - 1] - Ln[i - 1]) == 0 else d / (Ln[d + i - 1] - Ln[i - 1])
    b = 0 if (Ln[d + i] - Ln[i]) == 0 else d / (Ln[d + i] - Ln[i])
    return a * BsFun(i, d - 1, t, Ln) - b * BsFun(i + 1, d - 1, t, Ln)


# Second derivative of B-spline
def BsFun_second_derivative(i, d, t, Ln):
    if d < 2:
        return 0.0
    a = (
        0
        if (Ln[d + i - 2] - Ln[i - 2]) == 0
        else d * (d - 1) / ((Ln[d + i - 2] - Ln[i - 2]) ** 2)
    )
    b = (
        0
        if (Ln[d + i - 1] - Ln[i - 1]) == 0
        else 2 * d * (d - 1) / ((Ln[d + i - 1] - Ln[i - 1]) ** 2)
    )
    c = 0 if (Ln[d + i] - Ln[i]) == 0 else d * (d - 1) / ((Ln[d + i] - Ln[i]) ** 2)
    return (
        a * BsFun(i, d - 2, t, Ln)
        - b * BsFun(i + 1, d - 2, t, Ln)
        + c * BsFun(i + 2, d - 2, t, Ln)
    )


def ground_truth(x_vals, T_vals, a, lambda_param):
    def f(x, t, a, lambda_param):
        if t == 0:
            return 0
        return (
            (a - x)
            / np.sqrt(2 * np.pi * t**3)
            * np.exp(-(((a - x) - lambda_param * t) ** 2) / (2 * t))
        )

    def F2(x, T, a, lambda_param):
        if x >= 2:  # already in safe set
            return 1
        else:
            result, _ = quad(
                lambda t: f(x, t, a, lambda_param), 0, T, epsabs=1e-7, epsrel=1e-7
            )
            return result

    F = torch.zeros((len(T_vals), len(x_vals)))
    for i, xi in enumerate(x_vals):
        for j, Tj in enumerate(T_vals):
            F[j, i] = F2(xi, Tj, a, lambda_param)

    return F


def chebyshev_distance(x, y, x0, y0):
    return np.maximum(np.abs(x - x0), np.abs(y - y0))


def gaussian_2d_chebyshev(x, y, x0, y0, sigma, max_distance):
    return np.exp(-chebyshev_distance(x, y, x0, y0) / (2 * sigma)) / np.exp(
        -max_distance / (2 * sigma)
    )


def create_array(N, M, t, max_t):
    # Define the center of the array
    x0, y0 = N - 1, 0  # // 2, M // 2

    # Calculate E(t) and S(t)
    E_t = 1 - t / (max_t)
    S_t = 2 * M * (1 - t / max_t) + 0.25 * M * (t / max_t)

    # Create coordinate grids
    x = np.linspace(0, N - 1, N)
    y = np.linspace(0, M - 1, M)
    X, Y = np.meshgrid(x, y)

    # Calculate the maximum Chebyshev distance
    max_distance = np.maximum(x0, N - 1 - x0) + np.maximum(y0, M - 1 - y0)

    # Calculate the Gaussian using Chebyshev distance
    G = gaussian_2d_chebyshev(X, Y, x0, y0, S_t, max_distance)
    edge_value = (G[x0, 0] + G[N - 1, y0]) / 2
    G = E_t * (G - edge_value) / (np.max(G) - edge_value)  # always 0 at edge
    # Create the final array
    result = 1 - G

    return result


# Compute derivatives of B-spline surfaces
def compute_bspline_derivatives(
    U_full,
    bit_t,
    bit_x,
    bit_t_derivative,
    bit_x_derivative,
    bit_t_second_derivative,
    bit_x_second_derivative,
):
    # First derivative with respect to time t (using first derivative of bit_t)
    B_surface_t = torch.matmul(torch.matmul(bit_t_derivative, U_full), bit_x.T)

    # First derivative with respect to space x (using first derivative of bit_x)
    B_surface_x = torch.matmul(torch.matmul(bit_t, U_full), bit_x_derivative.T)

    # Second derivative with respect to space x (using second derivative of bit_x)
    B_surface_xx = torch.matmul(torch.matmul(bit_t, U_full), bit_x_second_derivative.T)

    return B_surface_t, B_surface_x, B_surface_xx


def compute_bspline_derivatives_3d(
    U_full,
    bit_t,
    bit_x,
    bit_y,
    bit_z,
    bit_t_derivative,
    bit_x_derivative,
    bit_y_derivative,
    bit_z_derivative,
    bit_x_pp,
    bit_y_pp,
    bit_z_pp,
):

    # Jasmine's faster version of the code below.
    if U_full.ndim < 5:
        U_full = U_full.reshape(1, *U_full.shape)

    # Compute the surface and its derivatives
    B_surface = torch.einsum(
        "qijkl,ti,xj,yk,zl->qtxyz", U_full, bit_t, bit_x, bit_y, bit_z
    )
    B_surface_t = torch.einsum(
        "qijkl,ti,xj,yk,zl->qtxyz", U_full, bit_t_derivative, bit_x, bit_y, bit_z
    )
    B_surface_x = torch.einsum(
        "qijkl,ti,xj,yk,zl->qtxyz", U_full, bit_t, bit_x_derivative, bit_y, bit_z
    )
    B_surface_y = torch.einsum(
        "qijkl,ti,xj,yk,zl->qtxyz", U_full, bit_t, bit_x, bit_y_derivative, bit_z
    )
    B_surface_z = torch.einsum(
        "qijkl,ti,xj,yk,zl->qtxyz", U_full, bit_t, bit_x, bit_y, bit_z_derivative
    )
    B_surface_xx = torch.einsum(
        "qijkl,ti,xj,yk,zl->qtxyz", U_full, bit_t, bit_x_pp, bit_y, bit_z
    )
    B_surface_yy = torch.einsum(
        "qijkl,ti,xj,yk,zl->qtxyz", U_full, bit_t, bit_x, bit_y_pp, bit_z
    )
    B_surface_zz = torch.einsum(
        "qijkl,ti,xj,yk,zl->qtxyz", U_full, bit_t, bit_x, bit_y, bit_z_pp
    )

    # Laplacian (sum of second derivatives in each spatial direction)
    B_surface_laplacian = B_surface_xx + B_surface_yy + B_surface_zz

    return (
        B_surface,
        B_surface_t,
        B_surface_x,
        B_surface_y,
        B_surface_z,
        B_surface_laplacian,
    )


# Compute B-spline derivatives of the surface and apply weighting to the physics loss
def compute_weighted_physics_loss(
    U_full,
    bit_t,
    bit_x,
    bit_t_derivative,
    bit_x_derivative,
    bit_t_second_derivative,
    bit_x_second_derivative,
    lambda_train,
    x,
    t,
):
    # Compute B-spline derivatives (t, x, and x^2 derivatives)
    B_surface_t, B_surface_x, B_surface_xx = compute_bspline_derivatives(
        U_full,
        bit_t,
        bit_x,
        bit_t_derivative,
        bit_x_derivative,
        bit_t_second_derivative,
        bit_x_second_derivative,
    )

    # Compute the physics residuals based on the PDE: B_t = lambda * B_x + 0.5 * B_xx
    pde_residual = B_surface_t - lambda_train * B_surface_x - 0.5 * B_surface_xx

    # Generate the weighting matrix based on the spatial and temporal coordinates
    x_grid, t_grid = np.meshgrid(x, t)  # Create grid of x and t values
    weight_matrix = weight_function(
        x_grid, t_grid, x_target=2, T_target=0, scale_x=0.5, scale_T=0.5
    )  # Increase weight near (x=2, T=0)

    # Convert the weight matrix to a PyTorch tensor
    weight_tensor = torch.tensor(weight_matrix, dtype=torch.float32)

    # Apply the weight to the PDE residuals
    weighted_pde_residual = pde_residual * weight_tensor

    # Compute the weighted physics loss
    weighted_physics_loss = torch.norm(weighted_pde_residual)

    return weighted_physics_loss


# Define a weighting function that increases the weight near (x, T) = (2, 0)
def weight_function(x, T, x_target=2, T_target=0, scale_x=1.0, scale_T=1.0):
    """
    A Gaussian-like weight function that assigns higher weights near the target point (x_target, T_target).
    scale_x and scale_T control the sharpness of the weighting in the x and T dimensions.
    """
    return np.exp(
        -(
            (x - x_target) ** 2 / (2 * scale_x**2)
            + (T - T_target) ** 2 / (2 * scale_T**2)
        )
    )


def calculate_control_points(A, B, P):
    """Calculate the least square control points given matrices A, B, and P, ensuring correct dimensions."""
    try:
        # Check if dimensions align properly
        if A.shape[0] != P.shape[0]:
            raise ValueError(
                f"Dimension mismatch between A ({A.shape}) and P ({P.shape}). A's rows must match P's rows."
            )
        if P.shape[1] != B.shape[0]:
            raise ValueError(
                f"Dimension mismatch between P ({P.shape}) and B ({B.shape}). P's columns must match B's rows."
            )

        # Calculate the least squares solution
        # (A^T A)^{-1} A^T P
        intermediate_result = np.linalg.solve(A.T @ A, A.T @ P)

        # intermediate_result * B * (B^T B)^{-1}
        Q_hat = intermediate_result @ B @ np.linalg.inv(B.T @ B)

        return Q_hat
    except ValueError as e:
        print(f"Error in matrix dimensions: {e}")
        return None


def numerical_derivative(f, t, delta=1e-5):
    return (f(t + delta) - f(t - delta)) / (2 * delta)


def numerical_second_derivative(f, t, delta=1e-5):
    return (f(t + delta) - 2 * f(t) + f(t - delta)) / (delta**2)
