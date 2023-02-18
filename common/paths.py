import pathlib

ENTRYPOINTS_DIR = pathlib.Path(__file__).parent
ROOT_DIR = ENTRYPOINTS_DIR.parent
BROWSER_SPECTATOR_DIR = ROOT_DIR / "browser-spectator"
SNAKES_DIR = ROOT_DIR / "snakes"
BIN_DIR = ROOT_DIR / "bin"