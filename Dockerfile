FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV GIT_PYTHON_REFRESH=quiet

ENTRYPOINT ["python", "-m", "trading_agent.cli"]
CMD ["--help"]
