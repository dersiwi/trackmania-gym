import argparse
import logging
from pathlib import Path
import os



def get_argparser() -> any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmi_port", "-p", type=int, default=8775)
    parser.add_argument("--launch", "-l", action="store_true",  default=False)
    parser.add_argument("--linux", "-u", action="store_true",  default=False)
    parser.add_argument("--reqimgs", "-imgs", action="store_true",  help="If set, requests images each simulation step and stores them in the current directory in a /frame folder. [WARNING : This is a ton of frames, even for short amounts of running it.]", default=False)
    parser.add_argument("--replay", "-r", type=str, help="Specifies the path to an action_log-file and replays it.", default=None)
    args = parser.parse_args()
    return args


def config_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/app.log"),
            logging.StreamHandler()
        ]
    )

def get_paths(linux : bool = False) -> tuple[str, str]:
    """Return paths to tm-loader and to plugin"""
    if linux:
        home_dir = os.environ['HOME']
        tmloader = Path(home_dir) / ".wine" / "drive_c" / "Program_Files_x86" / "TmNationsForever" / "TMLoader.exe"
        plugin = Path(home_dir) / "Documents" / "TMInterface" /  "Plugins" / "Python_Link.as"
    else:
        tmloader = Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
        plugin = Path(os.path.expanduser("~")) / "OneDrive" / "Dokumente" / "TMInterface" /"Plugins" / "Python_Link.as"
    return tmloader, plugin