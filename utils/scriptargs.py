import argparse

def get_argparser() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmi_port", "-p", type=int, default=8775)
    parser.add_argument("--launch", "-l", action="store_true",  default=False)
    parser.add_argument("--linux", "-u", action="store_true",  default=False)
    parser.add_argument("--reqimgs", "-imgs", action="store_true",  help="If set, requests images each simulation step and stores them in the current directory in a /frame folder. [WARNING : This is a ton of frames, even for short amounts of running it.]", default=False)

    args = parser.parse_args()
    return args