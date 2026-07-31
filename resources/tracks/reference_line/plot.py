import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
import os
# Load the data
ap = argparse.ArgumentParser()
ap.add_argument("map", type=str, help="Name of the map. Assumes map is in reference_line/ folder. Add .npy extension.")
args = ap.parse_args()

reflines_dir = os.path.dirname(os.path.abspath(__file__))
refline_path = os.path.join(reflines_dir, args.map)
assert os.path.exists(refline_path), f"Looking for reference-line-file '{args.map}' in directory '{reflines_dir}'. Ful path : '{refline_path}'." 
data: np.ndarray = np.load(refline_path)

x = data[:, 0]
y = data[:, 2]
z = data[:, 1]



# Normalize z-values for colormap
norm = plt.Normalize(vmin=z.min(), vmax=z.max())
cmap = plt.cm.plasma
colors = cmap(norm(z))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=colors, s=10)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D Reference Line Colored by Height (Z)")


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    max_span = max(spans)
    new_limits = np.array([centers - max_span / 2, centers + max_span / 2]).T
    ax.set_xlim3d(new_limits[0])
    ax.set_ylim3d(new_limits[1])
    ax.set_zlim3d(new_limits[2])

set_axes_equal(ax)

# Add colorbar (legend for z)
mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array([])  # Required for colorbar
cbar = fig.colorbar(mappable, ax=ax, pad=0.1)
cbar.set_label("Z-Height (Color Legend)")

plt.show()
