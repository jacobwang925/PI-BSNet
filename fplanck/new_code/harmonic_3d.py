# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from fplanck import fokker_planck, boundary, gaussian_pdf, harmonic_potential
# from mpl_toolkits.mplot3d import Axes3D

# nm = 1e-9
# viscosity = 8e-4
# radius = 50 * nm
# drag = 6 * np.pi * viscosity * radius

# # 3D harmonic potential centered at (0, 0, 0)
# U = harmonic_potential((0, 0, 0), 1e-6)

# # Extend to 3D with an extent of [x_range, y_range, z_range]
# sim = fokker_planck(
#     temperature=300,
#     drag=drag,
#     extent=[600 * nm, 600 * nm, 600 * nm],
#     resolution=10 * nm,
#     boundary=boundary.reflecting,
#     potential=U,
# )

# # Initial condition: 3D Gaussian distribution centered at (-150, -150, -150) nm
# pdf = gaussian_pdf(center=(-150 * nm, -150 * nm, -150 * nm), width=30 * nm)
# p0 = pdf(*sim.grid)

# Nsteps = 200
# time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)

# # Plotting 3D
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), constrained_layout=True)

# # Select the middle z slice to plot a 2D slice (x, y plane)
# mid_slice = int(p0.shape[2] / 2)
# X, Y = np.meshgrid(sim.grid[0][:, 0, 0] / nm, sim.grid[1][0, :, 0] / nm)  # Create 2D grid for x, y

# # Plot initial condition slice
# surf = ax.plot_surface(X, Y, p0[:, :, mid_slice], cmap="viridis")

# ax.set_zlim([0, np.max(Pt) / 3])
# ax.autoscale(False)

# # Update function for animation
# def update(i):
#     global surf
#     surf.remove()
#     # Plot PDF slice at z=mid_slice plane (mid-slice in z-dimension)
#     surf = ax.plot_surface(X, Y, Pt[i][:, :, mid_slice], cmap="viridis")

#     return [surf]

# anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
# ax.set(xlabel="x (nm)", ylabel="y (nm)", zlabel="normalized PDF")

# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, gaussian_pdf, harmonic_potential
from mpl_toolkits.mplot3d import Axes3D

nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius

# 3D harmonic potential centered at (0, 0, 0)
U = harmonic_potential((0, 0, 0), 1e-6)

# Extend to 3D with an extent of [x_range, y_range, z_range]
sim = fokker_planck(
    temperature=300,
    drag=drag,
    extent=[600 * nm, 600 * nm, 600 * nm],
    resolution=10 * nm,
    boundary=boundary.reflecting,
    potential=U,
)

# Initial condition: 3D Gaussian distribution centered at (-150, -150, -150) nm
pdf = gaussian_pdf(center=(-150 * nm, -150 * nm, -150 * nm), width=30 * nm)
p0 = pdf(*sim.grid)

Nsteps = 200
time, Pt = sim.propagate_interval(pdf, 2e-3, Nsteps=Nsteps)

# Indices for slicing the 3D data
mid_x = int(p0.shape[0] / 2)
mid_y = int(p0.shape[1] / 2)
mid_z = int(p0.shape[2] / 2)

# Create 2D grids for different planes
X_Y_grid = np.meshgrid(sim.grid[0][:, 0, 0] / nm, sim.grid[1][0, :, 0] / nm)  # X-Y plane
Y_Z_grid = np.meshgrid(sim.grid[1][0, :, 0] / nm, sim.grid[2][0, 0, :] / nm)  # Y-Z plane
X_Z_grid = np.meshgrid(sim.grid[0][:, 0, 0] / nm, sim.grid[2][0, 0, :] / nm)  # X-Z plane

# Plotting 3D slices for each plane
fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection='3d'), figsize=(18, 6))

# Plot initial condition slice for X-Y plane
X_Y_surf = axes[0].plot_surface(*X_Y_grid, p0[:, :, mid_z], cmap="viridis")
axes[0].set_title('X-Y plane')
axes[0].set_zlim([0, np.max(Pt) / 3])
axes[0].autoscale(False)

# Plot initial condition slice for Y-Z plane
Y_Z_surf = axes[1].plot_surface(*Y_Z_grid, p0[mid_x, :, :], cmap="plasma")
axes[1].set_title('Y-Z plane')
axes[1].set_zlim([0, np.max(Pt) / 3])
axes[1].autoscale(False)

# Plot initial condition slice for X-Z plane
X_Z_surf = axes[2].plot_surface(*X_Z_grid, p0[:, mid_y, :], cmap="inferno")
axes[2].set_title('X-Z plane')
axes[2].set_zlim([0, np.max(Pt) / 3])
axes[2].autoscale(False)

# Update function for animation
def update(i):
    global X_Y_surf, Y_Z_surf, X_Z_surf

    # Remove previous surfaces
    X_Y_surf.remove()
    Y_Z_surf.remove()
    X_Z_surf.remove()

    # Update X-Y plane
    X_Y_surf = axes[0].plot_surface(*X_Y_grid, Pt[i][:, :, mid_z], cmap="viridis")

    # Update Y-Z plane
    Y_Z_surf = axes[1].plot_surface(*Y_Z_grid, Pt[i][mid_x, :, :], cmap="plasma")

    # Update X-Z plane
    X_Z_surf = axes[2].plot_surface(*X_Z_grid, Pt[i][:, mid_y, :], cmap="inferno")

    return [X_Y_surf, Y_Z_surf, X_Z_surf]

# Animate the results
anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)

# Set labels for each plot
axes[0].set(xlabel="x (nm)", ylabel="y (nm)", zlabel="normalized PDF")
axes[1].set(xlabel="y (nm)", ylabel="z (nm)", zlabel="normalized PDF")
axes[2].set(xlabel="x (nm)", ylabel="z (nm)", zlabel="normalized PDF")

plt.show()
