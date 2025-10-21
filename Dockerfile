FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install pipenv and system dependencies needed for pipenv and Jupyter kernels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    uuid-runtime \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipfile and Pipfile.lock before installing dependencies (for better caching)
COPY Pipfile* /app/

# Install dependencies via pipenv --system to install into system Python (inside container)
RUN pipenv install --deploy --system

# Copy the rest of the application code (including notebooks)
COPY . .

# Expose port 8888 (optional; useful if you want to start notebook manually)
EXPOSE 8888

# Default CMD: open a shell (VS Code will handle launching Jupyter kernels)
CMD ["bash"]