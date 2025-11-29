# ---------------------------------------------------------
# Stage 1: Build React frontend
# ---------------------------------------------------------
FROM node:22-alpine AS frontend-build

WORKDIR /app/frontend

# Install dependencies
COPY paddy-assistant-web/package*.json ./
RUN npm install

# Copy the rest of the frontend source
COPY paddy-assistant-web/ .

# Build production bundle (e.g. Vite)
RUN npm run build


# ---------------------------------------------------------
# Stage 2: Python backend + built frontend
# ---------------------------------------------------------
FROM python:3.9-slim AS backend

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Optional system dependencies (build-essential helpful for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY chatbot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and CSV knowledge files
COPY chatbot/ ./

# Copy built frontend into /app/frontend
COPY --from=frontend-build /app/frontend/dist ./frontend

# Expose API port
EXPOSE 8000

# Start FastAPI (serves both API and static frontend)
CMD ["uvicorn", "chatbot_api:app", "--host", "0.0.0.0", "--port", "8000"]
