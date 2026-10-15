"""Lander Studio: train, watch and test the PPO lander live in the browser.

python -m lander.studio             # opens http://127.0.0.1:8765/
python -m lander.studio --no-open --port 8800
"""

import argparse

from lander.studio.server import serve


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--host", default="127.0.0.1", help="keep 127.0.0.1: the API can start jobs on this machine"
    )
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)
    serve(args.host, args.port, not args.no_open)


if __name__ == "__main__":
    main()
