FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8071
ENV FLASK_ENV=development
CMD ["python", "SKALD_server.py", "run", "--host=0.0.0.0", "--port=8071"]
