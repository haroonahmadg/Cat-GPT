# Cat GPT backend — Railway / Back4App / any Docker host
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps gTTS==2.5.4

COPY . .

EXPOSE 8000
CMD sh -c "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"
