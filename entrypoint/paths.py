import pathlib

ENTRYPOINT_DIR = pathlib.Path(__file__).parent
ROOT_DIR = ENTRYPOINT_DIR.parent
BROWSER_SPECTATOR_DIR = ROOT_DIR / "browser-spectator"
SNAKES_DIR = ROOT_DIR / "snakes"
BIN_DIR = ROOT_DIR / "bin"