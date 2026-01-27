# Journal Monitor

A web application that monitors journal updates (via RSS and Crossref) and sends push notifications to your phone via [Bark](https://github.com/Finb/Bark).

## Features

- **Multiple Sources**: Support for RSS/Atom feeds and Crossref API
- **Smart Deduplication**: Avoid duplicate notifications using DOI or content fingerprinting
- **Scheduled Checks**: Configurable daily check time
- **Push Notifications**: Send updates to your phone via Bark
- **Web Interface**: Manage subscriptions, view entries, and configure settings
- **Web-based Configuration**: All settings can be modified through the web UI

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. First-time Setup

On first startup, the application will:
1. Create the database automatically
2. Generate a random admin password
3. Print the password to the console logs

**Look for this in the startup logs:**

```
============================================================
FIRST TIME SETUP - ADMIN CREDENTIALS
============================================================
Username: admin
Password: <randomly-generated-password>
============================================================
Please save this password! It will not be shown again.
You can change it later in Settings > Reset Password.
============================================================
```

Visit `http://localhost:8000` and log in with the displayed credentials.

### 4. Configure via Web UI

After logging in, go to **Settings** to configure:

- **Bark Push Notification**: Enter your device key from the Bark app
- **Exa AI**: (Optional) API key for content extraction
- **Schedule**: Set check time and timezone
- **Push Behavior**: Configure notification grouping
- **HTTP Client**: Adjust timeout and User-Agent
- **Authentication**: Change username or password

## Configuration

All configuration is stored in the database and managed through the web interface. No `.env` file is required.

### Default Values

On first startup, the following defaults are used:

| Setting | Default |
|---------|---------|
| Check Time | 08:00 |
| Timezone | Asia/Shanghai |
| Bark Server | https://api.day.app |
| Request Timeout | 30 seconds |
| Merge Notifications | Yes |
| Max Entries per Message | 10 |

You can modify any of these settings through the web UI at `/settings`.

## API Endpoints

All API endpoints require authentication (except `/healthz` and `/api/entries`).

- `GET /healthz` - Health check
- `GET /` - Dashboard (requires login)
- `GET /entries` - View entries (public)
- `GET /settings` - Settings page (requires login)
- `GET /api/settings` - Get current settings
- `PUT /api/settings` - Update settings
- `POST /api/settings/password` - Set new password
- `POST /api/settings/password/rotate` - Generate random password
- `GET /api/subscriptions` - List all subscriptions
- `POST /api/subscriptions` - Create a new subscription
- `DELETE /api/subscriptions/{id}` - Delete a subscription
- `POST /api/check/run` - Manually trigger a check
- `POST /api/push/test` - Test Bark push notification
- `GET /api/entries` - List recent entries (public)
- `GET /api/runs` - List check run history

## Subscription Types

### RSS/Atom Feed

Add any journal's RSS feed URL. The monitor will check for new entries.

### Crossref

Add a journal by its ISSN. The monitor will query Crossref API for new publications.

## Project Structure

```
journal-monitor/
├── app/
│   ├── main.py          # FastAPI application entry
│   ├── config.py        # Configuration models and cache
│   ├── config_store.py  # Database config persistence
│   ├── db.py            # Database setup
│   ├── models.py        # SQLAlchemy models
│   ├── scheduler.py     # APScheduler setup
│   ├── runner.py        # Check runner logic
│   ├── exa_ai.py        # Exa AI content extraction
│   ├── sources/
│   │   ├── base.py      # Base source class
│   │   ├── rss.py       # RSS source implementation
│   │   └── crossref.py  # Crossref source implementation
│   ├── notifier/
│   │   └── bark.py      # Bark push notification
│   └── web/
│       ├── routes.py    # Web page routes
│       ├── api.py       # API routes
│       ├── auth.py      # Authentication helpers
│       ├── templates/   # Jinja2 templates
│       └── static/      # Static files (CSS, JS)
├── data/                # SQLite database (auto-created)
├── requirements.txt
└── README.md
```

## Example Subscriptions

Here are some example journal RSS feeds and ISSNs you can use:

### RSS Feeds

| Journal | Feed URL |
|---------|----------|
| Nature | `https://www.nature.com/nature.rss` |
| Science | `https://www.science.org/rss/news_current.xml` |
| Cell | `https://www.cell.com/cell/rss/current` |
| PNAS | `https://www.pnas.org/rss/current.xml` |
| PLoS ONE | `https://journals.plos.org/plosone/feed/atom` |
| arXiv (cs.AI) | `http://export.arxiv.org/rss/cs.AI` |

### Crossref (ISSN)

| Journal | ISSN |
|---------|------|
| Nature | `0028-0836` |
| Science | `0036-8075` |
| Cell | `0092-8674` |
| PNAS | `0027-8424` |
| Nature Medicine | `1078-8956` |
| New England Journal of Medicine | `0028-4793` |

### Adding via Web UI

1. Log in to the application
2. Go to **Subscriptions**
3. Click **Add Subscription**
4. Select source type (RSS or Crossref)
5. Enter the feed URL or ISSN
6. Click **Add**

## Troubleshooting

### Common Issues

1. **Forgot admin password**
   - Recommended: reset from CLI (keeps your data)
     - Stop the app
     - Run `python -m app.cli reset-password`
     - Restart the app and log in with the new password shown
   - Alternative (destructive): delete local data to regenerate credentials
     - Delete the database file (`data/journal_monitor.sqlite3`)
     - Delete the session secret file (`data/.session_secret`)
     - Restart the application to generate new credentials

2. **Bark notifications not working**
   - Go to Settings and enter your Bark device key
   - Test with the "Send Test Notification" button
   - Check the Bark app is properly configured on your phone

3. **RSS feed not updating**
   - Some feeds have caching, updates may take time
   - Check the feed URL is accessible in your browser
   - Look at the logs for any HTTP errors

4. **Crossref returning empty results**
   - Verify the ISSN is correct (format: `1234-5678`)
   - Try searching on [crossref.org](https://search.crossref.org/) to confirm
   - The journal must be registered with Crossref

### Logs

The application logs to stdout. Check the console output for errors and status messages.

## Security Notes

- Admin password is hashed using PBKDF2-SHA256 with 600,000 iterations
- Session cookies are signed with a randomly generated secret
- Sensitive API keys (Bark, Exa) are stored in the database (not logged)
- Authentication is always enabled - there is no anonymous admin mode

## License

MIT
