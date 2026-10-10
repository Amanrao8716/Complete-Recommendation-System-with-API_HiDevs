FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////app/recsys.db \
    PORT=5000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
# Seed the database on first start, then serve with gunicorn.
CMD ["sh", "-c", "python scripts/seed_data.py --if-empty && gunicorn -w 1 --threads 8 -b 0.0.0.0:${PORT} 'api.app:create_app()'"]
