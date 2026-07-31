from trackmania_gym.plotting.plotter import EnvPlotter
import matplotlib.pyplot as plt

class LinePlotter(EnvPlotter):
    """
    A class to efficiently plot multiple lines from a data dictionary
    """
    def __init__(self, keys_to_plot:list[str], title:str, ylabel:str, xlabel:str = "Step", ylim:tuple[int,int]=(0,1),xlim:tuple[int,int]=(0,40)):

        #plt.ion()  # Turn on interactive mode
        self.keys_to_plot = keys_to_plot

        self.fig, self.ax = plt.subplots(figsize=(8, 6)) 
        self.ax.set_title(title)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        
        self.vals = {}  # Dictionary to store the history of each line
        self.lines = {}  # Dictionary to store Line2D objects
        for k in keys_to_plot:
            self.vals[k] = []
            line, = self.ax.plot([], [], label=k, animated=True)
            self.lines[k] = line 

        self.ylim = ylim
        if self.ylim:
            self.ax.set_ylim(self.ylim)

        self.xlim = xlim
        if self.xlim:
            self.ax.set_xlim(self.xlim)
            
        # Draw the canvas once and capture the background so we can avoid redrawing it
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        
        plt.show(block=False)

    def setup_plot(self):
        return super().setup_plot()

    def plot(self, data: dict):

        for key, value in data.items():
            self.vals[key].append(value)
            self.lines[key].set_data(range(len(self.vals[key])), self.vals[key])

        #Check if axes limits need to change
        needs_hard_redraw = False  
        old_ylim = self.ax.get_ylim() 
        old_xlim = self.ax.get_xlim()
        # print(f"Old limits: x={old_xlim}, y={old_ylim}") 

        self.ax.relim()
        # self.ax.autoscale()

        # we only consider plots that expand to the right
        new_data_xlim = self.ax.dataLim.intervalx
        if new_data_xlim[1] > old_xlim[1]:
            num_points = len(next(iter(self.vals.values())))
            self.ax.set_xlim((old_xlim[0],self.ax.dataLim.intervalx[1]+num_points//3)) 
            needs_hard_redraw = True

        new_data_ylim = self.ax.dataLim.intervaly
        y_range = new_data_ylim[1] - new_data_ylim[0]
        y_buffer = y_range * 0.2
        new_ylim_lower,new_ylim_upper = old_ylim 

        if new_data_ylim[1] > old_ylim[1] :
            new_ylim_upper = new_data_ylim[1] + y_buffer 
            needs_hard_redraw = True

        if new_data_ylim[0] < old_ylim[0]:
            new_ylim_lower = new_data_ylim[0] - y_buffer
            needs_hard_redraw = True

        self.ax.set_ylim(new_ylim_lower,new_ylim_upper)

        # Full redraw is needed axes limits changed
        if needs_hard_redraw:
            self.ax.legend()
            # print(f"New limits: x={self.ax.get_xlim()}, y={self.ax.get_ylim()}")
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        else:
            self.fig.canvas.restore_region(self.background)
            # Draw only the the lines
            for line in self.lines.values():
                self.ax.draw_artist(line)
            # Blit the updated artists onto the canvas
            self.fig.canvas.blit(self.ax.bbox)

        self.fig.canvas.flush_events()
