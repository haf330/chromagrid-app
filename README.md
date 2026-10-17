---
title: Chromagrid Assistant
emoji: 🧱
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# Chromagrid Assistant

Notebook and app runtime for the Agentic Systems assignment. This fictional business assistant
provides product guidance and quotes, records feedback, and saves customer enquiries
only after confirmation of the displayed details. It does not place orders, accept
payment, notify staff, or guarantee follow-up.

## Demonstration video

[Watch the chatbot demonstration](Chromagrid_demo.mp4).

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
Locally, the server listens on 127.0.0.1:7862. On Hugging Face Spaces, it uses
0.0.0.0:7860 automatically when SPACE_ID is present. No Gradio share tunnel is used.

## Run the assignment notebook

From the activated virtual environment:

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name chromagrid --display-name "Chromagrid"
jupyter lab business_agent.ipynb
```

Select the **Chromagrid** kernel. Run the cells in order. Offline checks and UI
construction do not require a key or make paid calls. Set `LAUNCH_APP = True` in
the final code cell to launch the chatbot. Sending messages uses the configured key.
For Colab, clone or upload this repository, set `PROJECT_ROOT`, install dependencies,
and configure `OPENAI_API_KEY` in Colab secrets.

## Files

- `business_agent.ipynb`: main assignment notebook with labeled persona/instructions, routing, tools, memory, reflection, offline checks, and UI.
- `app.py`: loads settings and starts the Gradio application.
- `chromagrid/`: shared chat, routing, model, memory, tools, consent, and reflection code.
- `me/about_business.pdf` and `me/business_summary.txt`: required business references.
- `requirements.txt`: runtime dependencies.
- `.env.example`: credential and model configuration placeholders.

The app creates files under `logs/` when records are written. These files persist
locally across restarts, but chat sessions are not restored. The logs and credentials
are excluded from Git. Assignment diagnostics expose routing, tools, usage, and
reflection results during a chat.