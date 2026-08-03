ANSI_RESET = "\033[0m"

ANSI_BLUE = "\033[94m"
ANSI_GREEN = "\033[92m"
ANSI_CYAN = "\033[96m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_LIGHT_BLUE = "\033[94m"

def color(text: str, ansi: str) -> str:
    return f"{ansi}{text}{ANSI_RESET}"