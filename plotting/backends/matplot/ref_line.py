from plotting.core import EnvPlotter
import matplotlib.pyplot as plt
import numpy as np

class Plot_RefLine(EnvPlotter):

    def __init__(self, reference_line : np.ndarray):
        super().__init__()

        self.reference_line = reference_line

        self.fig, self.ax = plt.subplots(figsize=(11, 11))
        self.ax.set_title("XZ View (World Space)")
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Z Axis")
        self.ax.set_aspect('equal')

        # Static Reference Line (full track)
        self.ax.plot(
            -1. * self.reference_line[:, 0], 
            self.reference_line[:, 2], 
            color='black', 
            linestyle='-', 
            label='Full Reference Line'  # Label for the legend
        )
        
        # Incoming Reference Points 
        self.incoming_ref_points, = self.ax.plot(
            [], [], 
            color='red', 
            marker='o', 
            linestyle='-',
            animated = True,
            label='Upcoming Ref. Points' 
        )
        
        # Car Position 
        self.pos = self.ax.scatter(
            -1. * self.reference_line[0,0], 
            self.reference_line[0,2], 
            color='blue', 
            marker='D', 
            s=80, 
            animated = True,
            label='Car Position' 
        )
        
        # Next Closest Reference Point 
        self.ref_point = self.ax.scatter(
            -1. * self.reference_line[0,0], 
            self.reference_line[0,2], 
            color='orange', 
            marker='*', 
            s=120, 
            animated = True,
            label='Next Ref. Point' 
        )
        
        self.ax.legend(loc='best') 
        
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        plt.show(block=False)

    def plot(self,info):
        
        points = info["comming_refline_points"]
        orientation = info["orientation"]
        inv_orientation = orientation.T
        position = info["position"]
        next_refline_index = info["next_refline_index"]

        # 1. Check determinant ≈ 1 since in_rotation should be a rotation matrix 
        det = np.linalg.det(inv_orientation)
        assert np.isclose(det, 1.0, atol=1e-5), f"Determinant not close to 1: det = {det}"
        # 2. Check if inv_orientation @ orientation ≈ Identity
        ident = inv_orientation @ orientation
        assert np.allclose(ident, np.eye(3), atol=1e-5), f"Matrix product not identity:\n{ident}"

        # Transform points to world coordinates
        points = (inv_orientation @ points.T).T + position

        self.incoming_ref_points.set_data(-1. * points[:, 0], points[:, 2])
        self.pos.set_offsets([-1. * position[0], position[2]])
        self.ref_point.set_offsets([-1. * self.reference_line[next_refline_index, 0], self.reference_line[next_refline_index, 2]])
        
        needs_hard_redraw = False
        if needs_hard_redraw:
            pass
        else:
            self.fig.canvas.restore_region(self.background)
            self.ax.draw_artist(self.incoming_ref_points)
            self.ax.draw_artist(self.pos)
            self.ax.draw_artist(self.ref_point)
            self.fig.canvas.blit(self.ax.bbox)
        
        self.fig.canvas.flush_events()

    def setup_plot(self):
        return super().setup_plot()
