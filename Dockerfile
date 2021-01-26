FROM python:3.8-slim-buster

COPY requirements.txt /requirements.txt

RUN pip3 install -r requirements.txt

COPY ./api /api/api

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0"]
