# Repository Guidelines

## Project Structure & Module Organization
The application lives in `app/` and is a FastAPI service with a small web UI. Key areas:
- `app/main.py`: FastAPI entry point.
- `app/web/`: web pages (`templates/`) and API endpoints (`api.py`, `routes.py`).
- `app/sources/`: RSS and Crossref source implementations.
- `app/notifier/`: Bark push notification integration.
- `data/`: SQLite database files (auto-created at runtime).

## Build, Test, and Development Commands
Use a virtual environment and install dependencies:
- `python -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

Run the app:
- `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` (dev)
- `uvicorn app.main:app --host 0.0.0.0 --port 8000` (prod)

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
- Secrets (Bark/Exa keys) should never be logged.
- First-time startup prints admin credentials to the console—handle with care.
