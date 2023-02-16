import pathlib

ENTRYPOINT_DIR = pathlib.Path(__file__).parent
ROOT_DIR = ENTRYPOINT_DIR.parent
BOARD_DIR = ROOT_DIR / "board"
SNAKES_DIR = ROOT_DIR / "snakes"
BIN_DIR = ROOT_DIR / "bin"