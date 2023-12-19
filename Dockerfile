FROM python:3.10.3-alphine

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN apk --no- add build-base
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

ENV HOST 0.0.0.0

EXPOSE 8080

CMD ["python", "main.py"]
