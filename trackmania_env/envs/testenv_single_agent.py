from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.utils.actionmap import ACTION_MAP, get_reverse_action_map
import keyboard
from typing import Callable
import time
from matplotlib import pyplot as plt
from pynput.keyboard import Key, Listener,KeyCode
import numpy as np
from trackmania_env.utils.reference_line_manager import ReferenceLineManager

class KEYS:
    """Enum for keys used in TestEnvironment."""
    UP = "nach-oben"
    DOWN = "nach-unten"
    LEFT = "nach-links"
    RIGHT = "nach-rechts"
    ESCAPE = "esc"
    SHIFT = "shift"

    @staticmethod
    def get_key_combo(left : bool, right : bool, accelerate : bool, brake : bool):
        """Translates the manual input of trackmania player into a string."""
        combostring = ""
        if left:
            combostring += KEYS.LEFT + " : "
        if right:
                combostring += KEYS.RIGHT + " : "

        if accelerate:
                combostring += KEYS.UP + " : "

        if brake:
            combostring += KEYS.DOWN + " : "
        return combostring
    


class TestEnvironmentCallback():
    """TestEnviornmentCallbacks are used to track, log, do whatever with data obtained by an environment per setp."""

    def __init__(self):
        self.n_step = 0
        """Counts environment-steps aka. how often _call_after_step was called."""

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        """This method is called by TestEnvironment.step_with_manual_input(), after everytime this method executes
        an environment step of the underlying environment."""
        pass

    def _call_after_run(self):
        """This method is called by TestEnvironment.step_with_manual_input(), after the main-loop has been executed via `esc`."""
        pass

class Live3dPlotEnvironmentCallback(TestEnvironmentCallback):

    def __init__(self):
        # Set up interactive plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Start the plot
        self._setup_plot()
        plt.ion()
        plt.show()

    def _setup_plot(self):
        """Responsible for settingup"""
        pass

class PrintRewardsToConsole(TestEnvironmentCallback):

    def __init__(self):
        super().__init__()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        if self.n_step % 128 == 0:
            for key in info["rewards"]:
                print(key, end=" | ")
            print("\n")
        for key in info["rewards"]:
            print(key,info["rewards"][key], end=" | ")
        print("\n")
        self.n_step += 1

class TestLinesightRewards(TestEnvironmentCallback):
    """Tacks vx, vy, vz and plots them after run."""
    
    def __init__(self):
        super().__init__()
        self.velocities = [[], [], []] #v_x, v_y, v_z

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        velocity = info["velocity"]
        for i in range(len(self.velocities)):
            self.velocities[i].append(velocity[i])


    def _call_after_run(self):
        time = range(len(self.v_x))  

        plt.figure(figsize=(12, 6))

        for idx, name, color in zip([1,2,3], ["v_x", "v_y", "v_z"], ["blue", "orange", "green"]):

            plt.subplot(3, 1, idx)
            plt.plot(time, self.velocities[idx-1], label=name, color=color)
            plt.ylabel(name)
            if idx == 3:
                plt.xlabel('Time Step')
            plt.grid(True)

        plt.tight_layout()
        plt.show()

class PrintRotation(Live3dPlotEnvironmentCallback):
    def __init__(self):
        self.quiver = None
        super().__init__()



    def _setup_plot(self):
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Live Rotation Matrix Axes")
        # Initial dummy arrows
        self.quiver = self.ax.quiver(0, 0, 0, 1, 0, 0, color='r', label="X-axis")
        self.quiver = self.ax.quiver(0, 0, 0, 0, 1, 0, color='g', label="Y-axis")
        self.quiver = self.ax.quiver(0, 0, 0, 0, 0, 1, color='b', label="Z-axis")
        self.ax.legend()

    def _update_plot(self, rot_matrix):
        # Clear and redraw
        self.ax.cla()
        self._setup_plot()

        origin = np.array([0, 0, 0])
        x_axis = rot_matrix[:, 0]
        y_axis = rot_matrix[:, 1]
        z_axis = rot_matrix[:, 2]

        self.ax.quiver(*origin, *x_axis, color='r', label="X-axis")
        self.ax.quiver(*origin, *y_axis, color='g', label="Y-axis")
        self.ax.quiver(*origin, *z_axis, color='b', label="Z-axis")
        plt.draw()
        plt.pause(0.001)

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        rot_matrix = np.array(info["rotation_matrix"])
        self._update_plot(rot_matrix)


class PrintVectorToNextReferencePoint(Live3dPlotEnvironmentCallback):
    def __init__(self):
        self.quiver = None
        super().__init__()



    def _setup_plot(self):
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Live Rotation Matrix Axes")
        # Initial dummy arrows
        self.quiver = self.ax.quiver(0, 0, 0, 1, 0, 0, color='r', label="X-axis")
        self.quiver = self.ax.quiver(0, 0, 0, 0, 1, 0, color='g', label="Y-axis")
        self.quiver = self.ax.quiver(0, 0, 0, 0, 0, 1, color='b', label="Z-axis")
        self.ax.legend()



    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        
        self.ax.cla()
        self._setup_plot()

        origin = np.array([0, 0, 0])

        self.ax.quiver(*origin, *info["comming_refline_points"][0], color='r', label="X-axis")
        self.fig.canvas.draw()
        plt.pause(0.001)

class PrintVector2DToNextReferencePoint(TestEnvironmentCallback):
    def __init__(self):
        super().__init__()

        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self._setup_plot()

        plt.ion()
        plt.show()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        self.ax.cla()  # Clear previous plot
        self._setup_plot()  # Reset labels/aspect

        points :np.ndarray= info["comming_refline_points"]

        self.ax.scatter(points[0, 0], points[0, 2], color='green', marker='x', s=50)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _call_after_run(self):
        plt.ioff()
        plt.show()

    def _setup_plot(self):
        self.ax.set_title("XZ-View (Car Coordinate System)")
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Z Axis")
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)


class Test_RefLine_Next_Point_Manager(TestEnvironmentCallback):
    def __init__(self, reference_line : np.ndarray):
        super().__init__()
        self.reference_line = reference_line
        # Set up a single 2D plot for the map
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self._setup_plot()

        plt.ion()
        plt.show()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        points = info["comming_refline_points"]
        inv_orientation = info["orientation"].T
        position = info["position"]

        next_refline_index = info["rewards"]["nextpoint_reference_index"]

        # 1. Check determinant ≈ 1 since in_rotation should be a rotation matrix 
        det = np.linalg.det(inv_orientation)
        assert np.isclose(det, 1.0, atol=1e-5), f"Determinant not close to 1: det = {det}"
        # 2. Check if inv_orientation @ orientation ≈ Identity
        ident = inv_orientation @ info["orientation"]
        assert np.allclose(ident, np.eye(3), atol=1e-5), f"Matrix product not identity:\n{ident}"

        # Transform points to world coordinates
        points = (inv_orientation @ points.T).T + position

        # Clear and redraw plot
        self.ax.cla()
        self._setup_plot()

        # Plot reference line (black), transformed points (red), and position (green), and next reference-line-point (blue)
        self.ax.plot(-1. * self.reference_line[:, 0], self.reference_line[:, 2], color='black', linestyle='-')
        self.ax.plot(-1. * points[:, 0], points[:, 2], color='red', marker='o', linestyle='-')
        self.ax.scatter(-1. * position[0], position[2], color='green', marker='x', s=50)
        self.ax.scatter(-1. * self.reference_line[next_refline_index, 0], self.reference_line[next_refline_index, 2], color='blue', marker='x', s=50)
        
        self.fig.canvas.draw()
        plt.pause(0.001)

    def _setup_plot(self):
        self.ax.set_title("XZ View (World Space)")
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Z Axis")
        self.ax.set_aspect('equal')

    """
    y_lim:
     - d = (0,2000)
     - drel = (0,1)
     - velocity_delta = (0,100)
    
    """
class Test_1D_Next_Point_Manager(TestEnvironmentCallback):
    def __init__(self, key_to_plot, y_lim=(-10, 10)):
        super().__init__()
        self.key_to_plot = key_to_plot
        self.y_lim = y_lim  # fixed y-axis scale
        self.vals = []

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self._setup_plot()

        plt.ion()
        plt.show()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        info["refline_idx"]  = info["rewards"]["nextpoint_reference_index"]
        assert self.key_to_plot in info, f"The Key '{self.key_to_plot}' is not in the info dict"
        val = info[self.key_to_plot]
        assert np.ndim(val) <= 1, f"The value for '{self.key_to_plot}' must be 1D, got shape {np.shape(val)}"

        self.vals.append(val)

        # Plot
        self.ax.cla()
        self._setup_plot()

        self.ax.plot(range(len(self.vals)),self.vals)

        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

    def _setup_plot(self):
        self.ax.set_title(f"Tracking '{self.key_to_plot}' Over Time")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel(self.key_to_plot)
        self.ax.set_ylim(*self.y_lim)  # fixed Y-axis limits

class Test_3D_Next_Point_Manager(TestEnvironmentCallback):
    def __init__(self, key_to_plot, y_lim=(-1, 1)):
        super().__init__()
        self.key_to_plot = key_to_plot
        self.y_lim = y_lim
        self.vals = []

        # Set up 3 vertically stacked subplots for x, y, z
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        self._setup_plot()

        plt.ion()
        plt.tight_layout()
        plt.show()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        assert self.key_to_plot in info, f"The key '{self.key_to_plot}' is not in the info dict"

        val = np.asarray(info[self.key_to_plot])
        assert val.ndim == 1 and val.shape[0] == 3, \
            f"The value for '{self.key_to_plot}' must be a 3D vector, got shape {val.shape}"

        self.vals.append(val)

        # Clear subplots
        for ax in self.axes:
            ax.cla()

        # Convert to array for plotting
        vals_array = np.array(self.vals)  # shape: (steps, 3)
        labels = ['x', 'y', 'z']
        colors = ['red', 'green', 'blue']

        for i in range(3):
            self.axes[i].plot(vals_array[:, i], color=colors[i])
            self.axes[i].set_ylabel(labels[i])
            self.axes[i].set_ylim(*self.y_lim)
            self.axes[i].grid(True)

        self.axes[2].set_xlabel("Step")
        self.fig.suptitle(f"3D Vector Components of '{self.key_to_plot}'")

        plt.draw()
        plt.pause(0.001)

    def _setup_plot(self):
        labels = ['x', 'y', 'z']
        for i, ax in enumerate(self.axes):
            ax.set_ylabel(labels[i])
            ax.set_ylim(*self.y_lim)
            ax.grid(True)
        self.axes[2].set_xlabel("Step")

class TestEnvironment(TMNF_Single_Agent_Env):

    def __init__(self, command_queue, response_queue, obs_manager, reward_calculator,reference_line, env_cfg,platform =  "windows"):
        super().__init__(command_queue, response_queue, obs_manager, reward_calculator, reference_line ,env_cfg=env_cfg)

        self.action_modifier : Callable = None
        self.step_while_doing_nothing = False
        """Variable for setp_with_manual_input. If not input was given, no (environment)-step is executed."""
        self.env_test_callback : list[TestEnvironmentCallback] = []

        self.keyboard = keyboard if platform == "windows" else LinuxKeyboardWrapper()

    def _action_modifier_drive_forward(self, action : int) -> int:
        """Action-modifier which is valid option to set for self.action_modifier."""
        return 0
    
    def add_env_test_calback(self, env_test_callback : TestEnvironmentCallback):
        """Adds testenvironment-callback used in self.step_with_manual_input()"""
        self.env_test_callback.append(env_test_callback)
    
    def set_action_modifier(self, action_modifier : Callable) -> None:
        self.action_modifier = action_modifier

    def step(self, action):
        """Calls super().step(action). If no self.action_modifier (e.g. self._action_moifier_drive_forawrd) is defined."""
        if not self.action_modifier == None:
            action = self.action_modifier(action)

        return super().step(action)
    
    def step_with_manual_input(self, time_between_actions : float = 0.012):
        """This method enables to send manual inputs to the TMNF_Single_Agent_Env; basically simulating you playing the game with extra steps.
        In order to do something with date coming from the environment after each-step, you can define your own TestEnvironmentCallback-class.
        Implement the methods _call_after_step(obs, rew, terminated, trucated, infos) and _call_after_run(). 
        Then, before callinth this method add them to this class via this.add_env_test_callback(). 

        After you start this method the main loop starts:
        ``` 
        while running:
            1. press key 
            2. complete environment step
            3. call _call_after_step for each added callback
        4. call _call_after_run
        ```
        The run ends, if you press `esc`.

        """
        REVERSE_ACTIONMAP = get_reverse_action_map()
        running = True
        no_actions_since_n_steps = 0 # indicates since how many steps no action was executed 
        while running:

            left, right, accelerate, brake = False, False, False, False

            if self.keyboard.is_pressed(KEYS.UP):
                accelerate = True

            if self.keyboard.is_pressed(KEYS.DOWN):
                brake = True

            if self.keyboard.is_pressed(KEYS.LEFT):
                left = True
            
            if self.keyboard.is_pressed(KEYS.RIGHT):
                right = True

            if self.keyboard.is_pressed(KEYS.ESCAPE):
                running = False

            if self.keyboard.is_pressed(KEYS.SHIFT):
                #super().random_reset()
                pass
            
            reverse_action = (left, right, accelerate, brake)
            try:
                action_index = REVERSE_ACTIONMAP[reverse_action]
            except KeyError:
                print(f"Invalid action; key-combination : {KEYS.get_key_combo(*reverse_action)}")

            no_actions_since_n_steps = 0 if any(reverse_action) else no_actions_since_n_steps + 1

            if not self.step_while_doing_nothing and not any(reverse_action) and no_actions_since_n_steps >= 2:
                continue
            
            obs, reward, terminated, truncated, info = self.step(action_index)
            if terminated or truncated:
                super().reset()

            for cb in self.env_test_callback:
                cb._call_after_step(obs, reward, terminated, truncated, info)

            time.sleep(time_between_actions)

        for cb in self.env_test_callback:
            cb._call_after_run()

class LinuxKeyboardWrapper:
    def __init__(self):
        self.key_map = {
            "nach-oben": Key.up,
            "nach-unten": Key.down,
            "nach-links": Key.left,
            "nach-rechts": Key.right,
            "esc": Key.esc,
            "shift": Key.shift
        }

        # Keep track of currently pressed keys
        self.pressed_keys = set()

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        self.pressed_keys.add(key)

    def on_release(self, key):
        self.pressed_keys.discard(key)

    def is_pressed(self, key_str):
        pynput_key = self.key_map.get(key_str)
        if pynput_key is None:
            raise ValueError(f"Key '{key_str}' is not mapped in LinuxKeyboardWrapper.")
        return pynput_key in self.pressed_keys