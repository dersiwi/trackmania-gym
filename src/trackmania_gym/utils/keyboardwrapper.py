
from pynput.keyboard import Key, Listener,KeyCode
import keyboard


class KEYS:
    """Enum for keys used in TestEnvironment."""
    UP = "nach-oben"
    DOWN = "nach-unten"
    LEFT = "nach-links"
    RIGHT = "nach-rechts"
    ESCAPE = "esc"
    SHIFT = "shift"
    RESET = "k"

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

class KeyboardWrapper:

    @staticmethod
    def get_keyboardmodule(platform : str):
        return  keyboard if platform == "windows" else KeyboardWrapper()

    def __init__(self):
        self.key_map = {
            "nach-oben": Key.up,
            "nach-unten": Key.down,
            "nach-links": Key.left,
            "nach-rechts": Key.right,
            "esc": Key.esc,
            "shift": Key.shift,
            "k" : KeyCode.from_char(KEYS.RESET)
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
