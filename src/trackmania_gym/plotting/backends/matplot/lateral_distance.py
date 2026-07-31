import numpy as np
from scipy.stats import norm
from trackmania_gym.plotting.plotter import EnvPlotter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes


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

        # Left side: Map View 
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_map.set_title("XZ View (World Space)")
        self.ax_map.set_xlabel("X Axis")
        self.ax_map.set_ylabel("Z Axis")
        self.ax_map.set_aspect('equal')
        self.ax_map.plot(-1. * self.reference_line[:, 0], self.reference_line[:, 2], color='black', label="Reference Line")
        # Create dynamic artists for the map view 
        refline_origin = self.reference_line[0]
        self.map_cur_pos, = self.ax_map.plot(-1*refline_origin[0], refline_origin[2], marker='x', color='green', markersize=10, label="Position", animated=True)
        self.map_cur_ref_pos, = self.ax_map.plot(-1*refline_origin[0], refline_origin[2], marker='x', color='blue', markersize=10, label="Reference Point", animated=True)
        self.map_arrow = None # Arrow will be recreated
        self.ax_map.legend(loc="upper left")

        # Top-right: Bar Plot 
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

        # Bottom-right: Gaussian PDF Plot 
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
        
        # Save initial backgrounds for blitting 
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
        
        # Update all dynamic artists 
        
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
        self.bar_container[0].set_height(lateral_distance) 
        self.bar_value_text.set_text(f"Value: {lateral_distance:.2f}") 

        # Gaussian Plot
        prob = self.gauss_dist.pdf(lateral_distance)
        self.gauss_vline.set_xdata([lateral_distance, lateral_distance])
        self.gauss_point.set_data([lateral_distance], [prob])
        
        # check for redraw/rescaling 
        map_xlim,map_ylim,need_hard_redraw = self._check_and_expand_limits(self.ax_map,100)
        if need_hard_redraw:
            self.ax_map.set_ylim(*map_ylim)
            self.ax_map.set_xlim(*map_xlim)
        old_ylim = self.ax_bar.get_ylim()
        new_ylim_upper = max(10, lateral_distance + 5) 
        if new_ylim_upper > old_ylim[1]:
            self.ax_bar.set_ylim(0, new_ylim_upper)
            need_hard_redraw = True

        if need_hard_redraw:
            # Full Redraw (Slower)
            self.fig.canvas.draw() 
            # Re-capture the new backgrounds with the updated axes
            self.map_bg = self.fig.canvas.copy_from_bbox(self.ax_map.bbox)
            self.bar_bg = self.fig.canvas.copy_from_bbox(self.ax_bar.bbox)
            self.gauss_bg = self.fig.canvas.copy_from_bbox(self.ax_gauss.bbox)
        else:
            # Fast Blitting 
            
            # Restore the backgrounds
            self.fig.canvas.restore_region(self.map_bg)
            self.fig.canvas.restore_region(self.bar_bg)
            self.fig.canvas.restore_region(self.gauss_bg)
            
            # Draw the dynamic artists over the restored background
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
            
            # Blit the regions onto the canvas
            self.fig.canvas.blit(self.ax_map.bbox)
            self.fig.canvas.blit(self.ax_bar.bbox)
            self.fig.canvas.blit(self.ax_gauss.bbox)

        self.fig.canvas.flush_events()

    def setup_plot(self):
        return super().setup_plot()

    def _check_and_expand_limits(self, ax: Axes, buffer: float = 1.0) -> tuple[list[float],list[float],bool]:
        ax.relim() 
        
        old_xlim = list(ax.get_xlim())
        old_ylim = list(ax.get_ylim())
        
        # dataLim reflects the bounding box of all artists
        new_data_xlim = ax.dataLim.intervalx
        new_data_ylim = ax.dataLim.intervaly
        
        new_xlim = list(old_xlim)
        new_ylim = list(old_ylim)
        need_redraw = False

        # Check X-limits (Left and Right)
        if new_data_xlim[1] > old_xlim[1]:
            new_xlim[1] = new_data_xlim[1] + buffer
            need_redraw = True
        if new_data_xlim[0] < old_xlim[0]:
            new_xlim[0] = new_data_xlim[0] - buffer
            need_redraw = True

        # Check Y-limits (Lower and Upper)
        if new_data_ylim[1] > old_ylim[1]:
            new_ylim[1] = new_data_ylim[1] + buffer
            need_redraw = True
        if new_data_ylim[0] < old_ylim[0]:
            new_ylim[0] = new_data_ylim[0] - buffer
            need_redraw = True

        return (new_xlim,new_ylim,need_redraw)

class Plot_Lateral_Distance2(Plot_Lateral_Distance):
    """Same as the normal Plot_Lateral_Distance but with only the map view."""
    def __init__(self, reference_line_manager):

        self.ref_line_manager = reference_line_manager
        self.reference_line = self.ref_line_manager.reference_line

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        self.ax.set_title("XZ View (World Space)")
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Z Axis")
        self.ax.set_aspect('equal')

        self.ax.plot(-1. * self.reference_line[:, 0],
                     self.reference_line[:, 2],
                     color='black', label="Reference Line")

        refline_origin = self.reference_line[0]
        self.map_cur_pos, = self.ax.plot(-1 * refline_origin[0],
                                         refline_origin[2],
                                         marker='x', color='green',
                                         markersize=10, label="Position",
                                         animated=True)
        self.map_cur_ref_pos, = self.ax.plot(-1 * refline_origin[0],
                                             refline_origin[2],
                                             marker='x', color='blue',
                                             markersize=10,
                                             label="Reference Point",
                                             animated=True)
        self.map_arrow = None  # Arrow will be recreated each frame

        self.lateral_text = self.ax.text(0.02, 0.5, "", transform=self.ax.transAxes,
                                         fontsize=12, color='red',
                                         ha='left', va='top', animated=True)

        self.ax.legend(loc="upper left")
        self.fig.tight_layout()
        self.fig.canvas.draw()

        self.map_bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        plt.show(block=False)

    def plot(self, info):
        position = info["position"]
        next_refline_index = info["next_refline_index"]
        ref_line_point = self.reference_line[next_refline_index]
        diff_vec = ref_line_point - position

        lateral_distance = self.ref_line_manager.calculate_lateral_difference(
            next_refline_index, position
        )

        self.map_cur_pos.set_data([-1 * position[0]], [position[2]])
        self.map_cur_ref_pos.set_data([-1 * ref_line_point[0]], [ref_line_point[2]])

        self.lateral_text.set_text(f"Lateral Distance: {lateral_distance:.2f} m")

        if self.map_arrow:
            self.map_arrow.remove()
        self.map_arrow = self.ax.arrow(-1. * position[0], position[2],
                                       -1. * diff_vec[0], diff_vec[2],
                                       head_width=0.5, head_length=1.0,
                                       fc='magenta', ec='magenta',
                                       length_includes_head=True, animated=True)

        # Check if we need to rescale
        map_xlim, map_ylim, need_hard_redraw = self._check_and_expand_limits(self.ax, 100)

        if need_hard_redraw:
            # Full redraw
            self.ax.set_xlim(*map_xlim)
            self.ax.set_ylim(*map_ylim)
            self.fig.canvas.draw()
            self.map_bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        else:
            self.fig.canvas.restore_region(self.map_bg)
            self.ax.draw_artist(self.map_cur_pos)
            self.ax.draw_artist(self.map_cur_ref_pos)
            self.ax.draw_artist(self.map_arrow)
            self.ax.draw_artist(self.lateral_text)
            self.fig.canvas.blit(self.ax.bbox)

        self.fig.canvas.flush_events()

class Plot_Lateral_Distance_MapAndGraph(Plot_Lateral_Distance):
    """
    A simplified Plot_Lateral_Distance that shows:
    - Left: Map view (XZ world space)
    - Right: Lateral distance as a function of time
    """

    def __init__(self, reference_line_manager):

        self.ref_line_manager = reference_line_manager
        self.reference_line = self.ref_line_manager.reference_line
        self.lateral_history = []

        self.fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_map.set_title("XZ View (World Space)")
        self.ax_map.set_xlabel("X Axis")
        self.ax_map.set_ylabel("Z Axis")
        self.ax_map.set_aspect('equal')

        self.ax_map.plot(-1. * self.reference_line[:, 0],
                         self.reference_line[:, 2],
                         color='black', label="Reference Line")

        refline_origin = self.reference_line[0]
        self.map_cur_pos, = self.ax_map.plot(-1 * refline_origin[0],
                                             refline_origin[2],
                                             marker='x', color='green',
                                             markersize=10, label="Position",
                                             animated=True)
        self.map_cur_ref_pos, = self.ax_map.plot(-1 * refline_origin[0],
                                                 refline_origin[2],
                                                 marker='x', color='blue',
                                                 markersize=10,
                                                 label="Reference Point",
                                                 animated=True)
        self.map_arrow = None
        self.ax_map.legend(loc="upper left")

        self.ax_graph = self.fig.add_subplot(gs[0, 1])
        self.ax_graph.set_title("Lateral Distance Over Time")
        self.ax_graph.set_xlabel("Step")
        self.ax_graph.set_ylabel("Lateral Distance (m)")

        self.graph_line, = self.ax_graph.plot([], [], color='magenta',
                                              label='Lateral Distance',
                                              animated=True)
        self.ax_graph.legend(loc="upper left")

        self.fig.tight_layout()
        self.fig.canvas.draw()

        self.map_bg = self.fig.canvas.copy_from_bbox(self.ax_map.bbox)
        self.graph_bg = self.fig.canvas.copy_from_bbox(self.ax_graph.bbox)

        plt.show(block=False)

    def plot(self, info):
        position = info["position"]
        next_refline_index = info["next_refline_index"]
        ref_line_point = self.reference_line[next_refline_index]
        diff_vec = ref_line_point - position

        lateral_distance = self.ref_line_manager.calculate_lateral_difference(
            next_refline_index, position
        )
        self.lateral_history.append(lateral_distance)

        if self.map_arrow:
            self.map_arrow.remove()
        self.map_cur_pos.set_data([-1 * position[0]], [position[2]])
        self.map_cur_ref_pos.set_data([-1 * ref_line_point[0]], [ref_line_point[2]])
        self.map_arrow = self.ax_map.arrow(-1. * position[0], position[2],
                                           -1. * diff_vec[0], diff_vec[2],
                                           head_width=0.5, head_length=1.0,
                                           fc='magenta', ec='magenta',
                                           length_includes_head=True,
                                           animated=True)

        x_vals = np.arange(len(self.lateral_history))
        self.graph_line.set_data(x_vals, self.lateral_history)

        # Auto-expand axes as needed
        need_redraw = False
        old_xlim = self.ax_graph.get_xlim()
        old_ylim = self.ax_graph.get_ylim()

        if len(x_vals) > 0 and x_vals[-1] > old_xlim[1]:
            self.ax_graph.set_xlim(0, x_vals[-1] + 150)
            need_redraw = True

        if max(self.lateral_history) > old_ylim[1]:
            self.ax_graph.set_ylim(0, max(self.lateral_history) + 5)
            need_redraw = True

        if need_redraw:
            self.fig.canvas.draw()
            self.map_bg = self.fig.canvas.copy_from_bbox(self.ax_map.bbox)
            self.graph_bg = self.fig.canvas.copy_from_bbox(self.ax_graph.bbox)
        else:
            self.fig.canvas.restore_region(self.map_bg)
            self.fig.canvas.restore_region(self.graph_bg)

            self.ax_map.draw_artist(self.map_cur_pos)
            self.ax_map.draw_artist(self.map_cur_ref_pos)
            self.ax_map.draw_artist(self.map_arrow)

            self.ax_graph.draw_artist(self.graph_line)

            self.fig.canvas.blit(self.ax_map.bbox)
            self.fig.canvas.blit(self.ax_graph.bbox)

        self.fig.canvas.flush_events()

