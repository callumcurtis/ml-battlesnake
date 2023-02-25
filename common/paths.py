import pathlib

COMMON_DIR = pathlib.Path(__file__).parent
ROOT_DIR = COMMON_DIR.parent
ENTRYPOINTS_DIR = ROOT_DIR / "entrypoints"
BROWSER_SPECTATOR_DIR = ROOT_DIR / "browser-spectator"
SNAKES_DIR = ROOT_DIR / "snakes"
BIN_DIR = ROOT_DIR / "bin"
ENGINE_DIR = ROOT_DIR / "engine"
ENVIROMENT_DIR = ROOT_DIR / "environment"
RULES_DIR = ROOT_DIR / "rules"
TEST_DIR = ROOT_DIR / "test"