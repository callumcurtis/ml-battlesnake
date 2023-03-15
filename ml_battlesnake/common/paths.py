import pathlib


COMMON_DIR = pathlib.Path(__file__).parent
ROOT_PACKAGE_DIR = COMMON_DIR.parent
ROOT_DIR = ROOT_PACKAGE_DIR.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = ROOT_DIR / "tests"
BIN_DIR = ROOT_DIR / "bin"
DEPLOYMENT_DIR = ROOT_PACKAGE_DIR / "deployment"
BROWSER_SPECTATOR_DIR = DEPLOYMENT_DIR / "browser-spectator"
GAME_ENGINE_AS_EXECUTABLE_DIR = DEPLOYMENT_DIR / "game-engine-as-executable"
SNAKES_DIR = DEPLOYMENT_DIR / "snakes"
LEARNING_DIR = ROOT_PACKAGE_DIR / "learning"
LEARNING_ENVIRONMENT_DIR = LEARNING_DIR / "environment"
GAME_ENGINE_AS_DLL_DIR = LEARNING_DIR / "game-engine-as-dll"