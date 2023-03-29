import subprocess


def main() -> None:
    subprocess.run(["pytest", "--picked"])
