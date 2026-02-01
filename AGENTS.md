# Repository Guidelines

## Project Structure & Module Organization
The application lives in `app/` and is a FastAPI service with a small web UI. Key areas:
- `app/main.py`: FastAPI entry point with lifespan management (DB init, scheduler start/stop).
- `app/config.py`: Configuration models (`RuntimeConfig`, `StaticConfig`) and in-process cache.
- `app/config_store.py`: Database-backed configuration persistence (replaces `.env` file).
- `app/db.py`: SQLAlchemy database setup and session management.
- `app/models.py`: SQLAlchemy ORM models (Subscription, Entry, Run, Config).
- `app/scheduler.py`: APScheduler integration for scheduled journal checks.
- `app/runner.py`: Core check runner logic that orchestrates source fetching and notifications. Sends a single digest notification per check run summarizing all journal updates.
- `app/exa_ai.py`: Optional Exa AI integration for web content extraction.
- `app/logging_config.py`: Application-wide logging configuration.
- `app/cli.py`: Command-line utilities (e.g., `reset-password`).
- `app/web/`: web pages (`templates/`) and API endpoints (`api.py`, `routes.py`, `auth.py`).
- `app/sources/`: RSS and Crossref source implementations (`base.py`, `rss.py`, `crossref.py`).
- `app/notifier/`: Bark push notification integration. Uses digest mode: one notification per check run with format "期刊更新：X 篇（Y 个期刊）", listing journals sorted by new entry count (descending).
- `data/`: SQLite database files (auto-created at runtime).
- `docs/how_to_deploy.md`: Deployment notes (CentOS/BT, systemd, data backup).
- `docs/sql_table.md`: SQLite table/schema reference with code pointers.
- `.cursor/`: Cursor IDE configuration (hooks, agents, rules) - gitignored, for development workflow.

## Documentation
- `docs/how_to_deploy.md`: Production-ish deployment checklist (Python venv, CentOS 7 gotchas, systemd service, reverse proxy notes, data backup).
- `docs/sql_table.md`: Database table overview for debugging data flow (table → feature → code location).

## Build, Test, and Development Commands
Use a virtual environment and install dependencies:
- `python -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

Run the app:
- `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` (dev)
- `uvicorn app.main:app --host 0.0.0.0 --port 8000` (prod)

CLI commands:
- `python -m app.cli reset-password`: Reset admin password (prints new password)
  - `--password <value>`: Set specific password (8-128 chars)
  - `--random`: Generate random password (default if no --password)

## Coding Style & Naming Conventions
- Python code uses 4-space indentation.
- Module and file names are `snake_case`.
- FastAPI routes and handlers live in `app/web/` with REST-style paths.
- Templates are Jinja2 in `app/web/templates/`, static assets in `app/web/static/`.

## Testing Guidelines
No test framework is set up in this repository yet. If you add tests, document how to run them and
follow a `tests/` directory layout.

## Commit & Pull Request Guidelines
This repository does not currently contain git history. If you initialize git, keep commit messages
short and action-oriented (e.g., "Add RSS retry handling"). For pull requests, include a brief
summary and a test plan or verification steps.

## Security & Configuration Notes
- Configuration is stored in the database; no `.env` is required.
- All runtime settings (Bark keys, Exa keys, schedule, HTTP settings) are managed via web UI at `/settings`.
- Configuration is cached in-process via `app/config.py`; changes require app restart to take effect.
- Secrets (Bark/Exa keys) should never be logged.
- First-time startup prints admin credentials to the console—handle with care and avoid pasting into docs/issues.
- Admin password can be reset via CLI: `python -m app.cli reset-password` (keeps data intact).
- If running under a process manager (e.g. systemd), restart the service after resetting password or changing config to ensure the running process uses the latest values.
- Session secrets are stored in `data/.session_secret` (auto-generated, gitignored).

## Notification Behavior
- Bark push notifications use digest mode: one notification per check run summarizing all journal updates.
- Notification format: title "期刊更新：X 篇（Y 个期刊）" with body listing journals sorted by new entry count (descending).
- Maximum number of journals shown in digest is controlled by `push_max_entries_per_message` setting (default: 10).
- The legacy `notify_new_entries()` method exists for backward compatibility but is no longer used by the main check runner.

## Deployment Notes (Ops)
- See `docs/how_to_deploy.md` for a concrete CentOS 7 + BT panel setup, including a sample systemd unit and data backup reminders.
- CentOS 7 may fail to build `greenlet` from source; install a binary wheel as described in `docs/how_to_deploy.md`.
