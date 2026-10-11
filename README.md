# Chromagrid Assistant

App runtime for the Agentic Systems assignment. This fictional business assistant
provides product guidance and quotes, records feedback, and saves customer enquiries
only after confirmation of the displayed details. It does not place orders, accept
payment, notify staff, or guarantee follow-up.

## Setup and run

Use Python 3.11 or later. Run these commands from this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start the app:

```bash
python app.py --port 7862
```

Open http://127.0.0.1:7862 in your browser. Messages can make paid OpenAI API calls.
The selected model is `gpt-6-luna`; account access and live behavior must be verified.
The server listens locally. This entry point is not a hosted deployment configuration.

## Files

- `app.py`: loads settings and starts the Gradio application.
- `chromagrid/`: shared chat, routing, model, memory, tools, consent, and reflection code.
- `me/about_business.pdf` and `me/business_summary.txt`: required business references.
- `requirements.txt`: runtime dependencies.
- `.env.example`: credential and model configuration placeholders.

The app creates files under `logs/` when records are written. These files persist
locally across restarts, but chat sessions are not restored. The logs and credentials
are excluded from Git. Assignment diagnostics expose routing, tools, usage, and
reflection results during a chat.

This repository is an app-only snapshot of the parent assignment project. The
notebook, tests, business authoring sources, generators, previews, and planning files
remain in that project. Parent-project changes do not automatically update this copy.
