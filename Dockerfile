FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HEADLESS=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY rl_sentiment_app.py .
COPY data/raw/VeiculosSubtraidos_2024.xlsx data/raw/VeiculosSubtraidos_2024.xlsx

RUN mkdir -p data/processed output

CMD ["python", "rl_sentiment_app.py"]
