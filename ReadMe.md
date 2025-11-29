# Paddy Leaf Disease Assistant

This project is a paddy (rice) disease assistant with:

- A **FastAPI** backend (`chatbot_api.py`) that diagnoses diseases from symptoms and provides tailored explanations and treatments using CSV knowledge bases.
- A **React** frontend that provides a ChatGPT-style chat interface.
- A **single Docker image** that bundles both backend and frontend so you can run everything with one container on macOS, Linux, or Windows.

---

## 1. Project Structure

Repository root (this folder):

```text
paddy-leaf-disease/
├── Dockerfile
├── chatbot/
│   ├── chatbot_api.py
│   ├── symptoms_causes.csv
│   ├── treatments_scenarios.csv
│   └── requirements.txt
└── paddy-assistant-web/
    ├── package.json
    ├── vite.config.* (or similar)
    └── src/ ...
```

> The backend expects the CSV files (`symptoms_causes.csv`, `treatments_scenarios.csv`) in the same folder as `chatbot_api.py`.

---

## 2. Quick Start (recommended): Run with Docker

This is the easiest, most portable way to run the app on **macOS, Windows, or Linux**.

### 2.1. Prerequisites

- **Docker Desktop** installed and running.
  - macOS: install via Homebrew (`brew install --cask docker`) or download from Docker’s website.
  - Windows: install Docker Desktop for Windows.

You do **not** need Python or Node installed on the host machine if you use Docker.

### 2.2. Build the Docker image

Open a terminal (or PowerShell on Windows), navigate to the project root (`paddy-leaf-disease`), and run:

```bash
cd path/to/paddy-leaf-disease

docker build -t paddy-assistant .
```

This will:

1. Build the React app in a Node build stage.
2. Build a Python image, install backend dependencies, add `chatbot_api.py` and CSVs.
3. Copy the built frontend into the backend image under `/app/frontend`.
4. Configure FastAPI to serve both the **API** and the **frontend** on port `8000`.

### 2.3. Run the container

```bash
docker run --rm -p 8000:8000 paddy-assistant
```

- `--rm` removes the container when you stop it.
- `-p 8000:8000` maps container port `8000` to host port `8000`.

### 2.4. Access the app

Open your browser and go to:

- **Web UI**: <http://localhost:8000/>
- **Health check**: <http://localhost:8000/api/health>

You should see:

- A chat interface titled **“Paddy Disease Assistant”**.
- The backend health endpoint returning JSON with `status: "ok"` and the list of diseases loaded.

To stop the app, press `Ctrl + C` in the terminal running `docker run`.

---

## 3. Running without Docker (local dev mode)

You only need this section if you want to develop or debug **outside** Docker.

### 3.1. Prerequisites

- Python 3.9+
- Node.js 18+ and npm

### 3.2. Backend (FastAPI)

From the project root:

```bash
cd chatbot

# Create and activate virtual environment (optional but recommended)
python -m venv .venv

# macOS / Linux:
source .venv/bin/activate

# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r requirements.txt
```

`requirements.txt` should include at least:

```text
fastapi
uvicorn[standard]
python-multipart
```

Run the backend:

```bash
uvicorn chatbot_api:app --reload --host 0.0.0.0 --port 8000
```

Check:

- <http://127.0.0.1:8000/api/health>

### 3.3. Frontend (React)

Open a new terminal, from the project root:

```bash
cd paddy-assistant-web

npm install
npm run dev
```

By default (Vite), the frontend runs at something like:

- <http://localhost:5173/>

The frontend should be configured with:

```ts
// In your React code (e.g. PaddyChat.tsx)
const API_URL = "http://127.0.0.1:8000/api/chat";
// or, if you use the same origin in Docker deployment:
// const API_URL = "/api/chat";
```

Make sure this matches the backend URL and port.

---

## 4. How the Chat Works

The chat flow:

1. User types:
   - Free-text symptom description, or
   - Follow-up questions like “How to treat it?”, “How to prevent it?”, “What causes this?”.
2. Frontend sends a `POST` request to `/api/chat` with:
   - `session_id`: a stable session id for the conversation.
   - `message`: current user message.
   - `history`: prior chat messages for context.
   - Optionally, a `cnn_prediction` if you integrate with an image classifier.
3. Backend:
   - Infers disease from symptoms or CNN prediction.
   - Explains symptoms, causes, conditions.
   - Provides treatment options and may set `awaiting_refinement = true` when it needs more details (weather, crop stage, water management).
4. User gives context like:
   - “Rainy weather, nursery stage, field is flooded”.
5. Backend:
   - Refines treatment recommendations based on that context and returns only the most relevant options.

Example `POST /api/chat` request (simplified):

```json
{
  "session_id": "sess_1234abcd",
  "message": "Brown lesions with yellow halo on leaves",
  "history": [
    { "role": "assistant", "content": "Hi, I’m your paddy disease assistant..." }
  ]
}
```

Example reply (simplified):

```json
{
  "session_id": "sess_1234abcd",
  "reply": "From the symptoms you described, this most likely matches brown spot...",
  "disease_name": "brown_spot",
  "intent": "GENERAL",
  "awaiting_refinement": false,
  "used_cnn_prediction": false,
  "debug": { }
}
```

---

## 5. Notes for Other Machines (Windows, Linux, macOS)

If you hand this project to someone else:

- **Recommend Docker**:
  - They only need Docker Desktop installed.
  - Instructions for them:
    ```bash
    cd paddy-leaf-disease
    docker build -t paddy-assistant .
    docker run --rm -p 8000:8000 paddy-assistant
    ```
  - Then open <http://localhost:8000/>.

- **No need for Python/Node** on their machine if they use Docker.

If they prefer to run locally without Docker, they can follow section 3 (Python + npm).

---

## 6. Troubleshooting

- **`docker: command not found`**  
  Docker is not installed or not on PATH. Install Docker Desktop first.

- **Frontend loads but chat fails**  
  - Check browser dev tools → Network → `/api/chat` requests.
  - Make sure the backend is running and `API_URL` matches (port and host).

- **CSV not found errors** in backend logs  
  - Ensure `symptoms_causes.csv` and `treatments_scenarios.csv` are inside `chatbot/` next to `chatbot_api.py`.
  - If you change file names or paths, update `chatbot_api.py` accordingly and rebuild the Docker image.

---

This should be enough for anyone (on macOS, Windows, or Linux) to build and run the Paddy Leaf Disease Assistant with either Docker or local dev tools.
