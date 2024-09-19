import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fplanck import fokker_planck, boundary, gaussian_pdf, gaussian_potential, combine
from mpl_toolkits.mplot3d import Axes3D

nm = 1e-9
viscosity = 8e-4
radius = 50 * nm
drag = 6 * np.pi * viscosity * radius

W = 60 * nm
A = 1.8e-20

# Combine multiple Gaussian potentials and a linear term in 3D
U = combine(
    gaussian_potential(center=(150 * nm, 150 * nm, 150 * nm), width=W, amplitude=A),
    gaussian_potential(center=(-150 * nm, -150 * nm, -150 * nm), width=W, amplitude=A),
    lambda x, y, z: -2e-14 * (x + y + z)
)

# Reduce extent and resolution for performance improvement
sim = fokker_planck(
    temperature=300,
    drag=drag,
    extent=[400 * nm, 400 * nm, 400 * nm],  # Reduced extent from 600nm to 400nm
    resolution=20 * nm,  # Increased resolution from 10nm to 20nm
    boundary=boundary.reflecting,
    potential=U
)

### Steady-state solution
steady = sim.steady_state()

### Time-evolved solution: 3D Gaussian PDF
pdf = gaussian_pdf(center=(-150 * nm, -150 * nm, -150 * nm), width=30 * nm)
p0 = pdf(*sim.grid)

# Reduce number of steps to optimize performance
Nsteps = 100  # Reduced from 200 to 100 steps
time, Pt = sim.propagate_interval(pdf, 0.1, Nsteps=Nsteps)

### Plotting 3D animation
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# We take a slice of the 3D grid for visualization (for example, at z = mid-point)
mid_z = int(p0.shape[2] / 2)
X, Y = np.meshgrid(sim.grid[0][:, 0, 0] / nm, sim.grid[1][0, :, 0] / nm)  # 2D grid for x, y

# Plot initial condition slice at z = mid-point
surf = ax.plot_surface(X, Y, p0[:, :, mid_z], cmap="viridis", alpha=0.7)

# Plot the steady-state solution (also a 2D slice at z = mid-point)
steady_surf = ax.plot_surface(X, Y, steady[:, :, mid_z], cmap="inferno", alpha=0.3)

ax.set_zlim([0, np.max(Pt) / 3])
ax.autoscale(False)

# Update function for animation
def update(i):
    global surf
    surf.remove()
    surf = ax.plot_surface(X, Y, Pt[i][:, :, mid_z], cmap="viridis", alpha=0.7)
    return [surf]

# Animate the solution
anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
ax.set(xlabel="x (nm)", ylabel="y (nm)", zlabel="normalized PDF")

plt.show()
