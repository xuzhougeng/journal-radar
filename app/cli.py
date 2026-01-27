"""
Command-line utilities for Journal Monitor.
"""

import argparse
import asyncio
import sys
from typing import Optional

from app.config import StaticConfig, get_runtime_config
from app.config_store import (
    ensure_config,
    generate_random_password,
    update_auth_password,
)
from app.db import async_session, init_db


def _validate_password(password: str) -> None:
    if not (8 <= len(password) <= 128):
        raise ValueError("Password length must be between 8 and 128 characters.")


async def _reset_password(password: Optional[str], use_random: bool) -> str:
    StaticConfig.ensure_data_dir()
    await init_db()

    async with async_session() as session:
        await ensure_config(session)

        if use_random or password is None:
            new_password = generate_random_password(16)
        else:
            _validate_password(password)
            new_password = password

        await update_auth_password(session, new_password)

    return new_password


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Journal Monitor CLI utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reset_parser = subparsers.add_parser(
        "reset-password",
        help="Reset admin password (prints the new password)",
    )
    reset_parser.add_argument(
        "--password",
        help="Set a specific password (8-128 characters)",
    )
    reset_parser.add_argument(
        "--random",
        action="store_true",
        help="Generate a random password (default if --password is not provided)",
    )

    args = parser.parse_args(argv)

    if args.command == "reset-password":
        if args.password and args.random:
            print("Error: --password and --random cannot be used together.", file=sys.stderr)
            return 2

        try:
            new_password = asyncio.run(_reset_password(args.password, args.random))
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        runtime_config = get_runtime_config()
        print("=" * 60)
        print("ADMIN PASSWORD RESET")
        print("=" * 60)
        print(f"Username: {runtime_config.auth_username}")
        print(f"Password: {new_password}")
        print("=" * 60)
        print("Restart the application to load the new password.")
        print("=" * 60)
        return 0

    print("Unknown command.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
