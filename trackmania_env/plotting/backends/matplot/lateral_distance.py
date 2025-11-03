import numpy as np
from scipy.stats import norm
from trackmania_env.plotting.core import EnvPlotter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from trackmania_env.utils.reference_line_manager import refline

class Plot_Lateral_Distance(EnvPlotter):

    def __init__(self, reference_line_manager):
        super().__init__()

        self.ref_line_manager = reference_line_manager
        self.reference_line = self.ref_line_manager.reference_line
        
        # Create values for the Gaussian curve
        self.gauss_dist = norm(0, np.sqrt(12)) # std dev = sqrt(12)
        self.x_vals = np.linspace(-50, 50, 500)
        self.y_vals = self.gauss_dist.pdf(self.x_vals)

        self.fig = plt.figure(figsize=(12, 8))
        
        # Use GridSpec to control layout
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

        # === Left side: Map View (ax_map) ===
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_map.set_title("XZ View (World Space)")
        self.ax_map.set_xlabel("X Axis")
        self.ax_map.set_ylabel("Z Axis")
        self.ax_map.set_aspect('equal')
        self.ax_map.plot(-1. * self.reference_line[:, 0], self.reference_line[:, 2], color='black', label="Reference Line")
        
        # Create dynamic artists for the map view (animated=True for blitting)
        refline_origin = self.reference_line[0]
        self.map_cur_pos, = self.ax_map.plot(-1*refline_origin[0], refline_origin[2], marker='x', color='green', markersize=10, label="Position", animated=True)
        self.map_cur_ref_pos, = self.ax_map.plot(-1*refline_origin[0], refline_origin[2], marker='x', color='blue', markersize=10, label="Reference Point", animated=True)
        self.map_arrow = None # Arrow will be recreated
        self.ax_map.legend(loc="upper left")

        # === Top-right: Bar Plot (ax_bar) ===
        self.ax_bar = self.fig.add_subplot(gs[0, 1])
        # Create the bar artist (animated=True)
        self.bar_container = self.ax_bar.bar(["Distance"], [0.0], color='magenta', animated=True)
        self.ax_bar.set_ylim(0, 10)
        self.ax_bar.set_ylabel("Euclidean Distance")
        self.ax_bar.set_title("Lateral Distance to Ref Point")
        
        # Create a dynamic text artist for the changing value (animated=True)
        self.bar_value_text = self.ax_bar.text(
            0.5, 
            0.9,
            f"Value: {0.0:.2f}",
            horizontalalignment='center', 
            transform=self.ax_bar.transAxes,
            animated=True,
            fontsize=10,
            color='black'
        )

        # === Bottom-right: Gaussian PDF Plot (ax_gauss) ===
        self.ax_gauss = self.fig.add_subplot(gs[1, 1])
        self.ax_gauss.plot(self.x_vals, self.y_vals, label='Gaussian PDF', color='blue')
        # Create dynamic artists (animated=True)
        self.gauss_vline = self.ax_gauss.axvline(0, color='red', linestyle='--', label='Current Distance', animated=True)
        self.gauss_point, = self.ax_gauss.plot(0, 0, 'ro', animated=True) 
        
        self.ax_gauss.set_title("Lateral Distance vs Gaussian PDF")
        self.ax_gauss.set_xlabel("Lateral Distance")
        self.ax_gauss.set_ylabel("Probability Density")
        self.ax_gauss.legend(loc="upper left")

        self.fig.tight_layout()
        self.fig.canvas.draw()
        
        # --- Save initial backgrounds for blitting ---
        self.map_bg = self.fig.canvas.copy_from_bbox(self.ax_map.bbox)
        self.bar_bg = self.fig.canvas.copy_from_bbox(self.ax_bar.bbox)
        self.gauss_bg = self.fig.canvas.copy_from_bbox(self.ax_gauss.bbox)

        plt.show(block=False)


    def plot(self, info):
        position = info["position"]
        next_refline_index = info["next_refline_index"]
        ref_line_point = self.reference_line[next_refline_index]
        diff_vec = ref_line_point - position
        lateral_distance = self.ref_line_manager.calculate_lateral_difference(next_refline_index, position)
        
        # --- Check for Axes Redraw ---
        old_ylim = self.ax_bar.get_ylim()
        new_ylim_upper = max(10, lateral_distance + 2) # Calculate required upper limit with buffer
        
        # Determine if a full redraw is needed (limits expand)
        need_hard_redraw = new_ylim_upper > old_ylim[1]
        
        # --- Update all dynamic artists ---
        
        # Map View
        if self.map_arrow:
            self.map_arrow.remove() # Remove old arrow before redraw/blit
            
        self.map_cur_pos.set_data([[-1 * position[0]], [position[2]]])
        self.map_cur_ref_pos.set_data([[-1. * ref_line_point[0]], [ref_line_point[2]]])
        
        # Arrow is recreated because it's hard to update. It is drawn later.
        self.map_arrow = self.ax_map.arrow(-1. * position[0], position[2], -1. * diff_vec[0], diff_vec[2],
                                           head_width=0.5, head_length=1.0, fc='magenta', ec='magenta', 
                                           length_includes_head=True, animated=True)

        # Bar Plot
        self.bar_container[0].set_height(lateral_distance) # Correctly updates the bar height
        self.bar_value_text.set_text(f"Value: {lateral_distance:.2f}") # Updates the fast text artist

        # Gaussian Plot
        prob = self.gauss_dist.pdf(lateral_distance)
        self.gauss_vline.set_xdata([lateral_distance, lateral_distance])
        self.gauss_point.set_data([lateral_distance], [prob])


        if need_hard_redraw:
            # Full Redraw (Slower, but necessary for axes change)
            
            # Apply new limits
            self.ax_bar.set_ylim(0, new_ylim_upper)
            
            # Full draw for everything (axes, grid, ticks, background elements)
            self.fig.canvas.draw() 
            
            # Re-capture the new backgrounds with the updated axes
            self.map_bg = self.fig.canvas.copy_from_bbox(self.ax_map.bbox)
            self.bar_bg = self.fig.canvas.copy_from_bbox(self.ax_bar.bbox)
            self.gauss_bg = self.fig.canvas.copy_from_bbox(self.ax_gauss.bbox)
            
        else:
            # Fast Blitting Logic (Axes limits are stable)
            
            # 1. Restore the backgrounds
            self.fig.canvas.restore_region(self.map_bg)
            self.fig.canvas.restore_region(self.bar_bg)
            self.fig.canvas.restore_region(self.gauss_bg)
            
            # 2. Draw the dynamic artists over the restored background
            # Map
            self.ax_map.draw_artist(self.map_cur_pos)
            self.ax_map.draw_artist(self.map_cur_ref_pos)
            self.ax_map.draw_artist(self.map_arrow)
            
            # Bar
            self.ax_bar.draw_artist(self.bar_container[0])
            self.ax_bar.draw_artist(self.bar_value_text) # Draw the fast-updating text
            
            # Gauss
            self.ax_gauss.draw_artist(self.gauss_vline)
            self.ax_gauss.draw_artist(self.gauss_point)
            
            # 3. Blit the regions onto the canvas
            self.fig.canvas.blit(self.ax_map.bbox)
            self.fig.canvas.blit(self.ax_bar.bbox)
            self.fig.canvas.blit(self.ax_gauss.bbox)

        self.fig.canvas.flush_events()

    def setup_plot(self):
        return super().setup_plot()
