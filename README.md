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
and configure `OPENAI_API_KEY` in Colab secrets. Colab execution remains unverified.

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

This repository is a submission runtime snapshot of the parent assignment project.
Tests, business authoring sources, generators, previews, and planning files remain
in that project. Parent-project changes do not automatically update this copy.

## Hugging Face Spaces deployment

1. Create a Space under your account, choose **Gradio**, and choose available CPU
   hardware. This app calls the OpenAI API and does not require a GPU. Check account
   eligibility and pricing before creating compute resources.
2. Upload the tracked repository files, preserving the `chromagrid/` and `me/`
   directories. Upload this README too: its YAML header selects Gradio 6.29.1,
   Python 3.12, and `app.py`. Do not upload `.env`, `.git/`, virtual environments,
   caches, or logs. The notebook is a submission deliverable; Spaces runs `app.py`.
3. In the Space's **Settings**, add a **Secret** named `OPENAI_API_KEY`. Paste your
   own key there. It is exposed to the server as an environment variable.
4. Wait for the build and inspect build/runtime logs if it fails. Once running,
   verify a product question, a quote, and a fictitious confirmation flow. Inspect
   Assignment diagnostics for actual routing, tool and reflection results.
5. Add the Space URL to your assignment submission and demonstration video.

A public Space lets other people use the app; model requests use your OpenAI account.
Pause the Space from Settings when you are finished demonstrating it. This app has
no automatic spending guard. The local USD 5 validation agreement is monitored
manually and is not a deployed application setting.

The default Space filesystem is ephemeral. JSONL records are demo logs and may be
lost when the container is rebuilt or replaced. Export any required demonstration
evidence before that happens; hosting does not add durable storage or chat recovery.

Official setup: https://huggingface.co/docs/hub/spaces-sdks-gradio
Secrets and account eligibility: https://huggingface.co/docs/hub/spaces-overview
Configuration: https://huggingface.co/docs/hub/spaces-config-reference
