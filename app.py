"""Local and Hugging Face Spaces entry point; credentials stay in environment/.env, never source."""

import argparse
import os
from pathlib import Path

from chromagrid.ui import build_app, chat_factory


def main():
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    on_spaces = bool(os.environ.get("SPACE_ID"))
    default_host = "0.0.0.0" if on_spaces else "127.0.0.1"
    default_port = 7860 if on_spaces else 7862
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()
    build_app(chat_factory(root)).queue().launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
