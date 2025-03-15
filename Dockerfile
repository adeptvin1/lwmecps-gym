FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ /app

ENV HOST=0.0.0.0
ENV PORT=8010

CMD [ "sleep", "5000" ]
