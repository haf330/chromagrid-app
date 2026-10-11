"""Local entry point; credentials stay in environment/.env, never source."""

import argparse
from pathlib import Path

from chromagrid.ui import build_app, chat_factory


def main():
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    build_app(chat_factory(root)).queue().launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
