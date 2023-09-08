FROM debian:buster-slim
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && pip3 install flask
WORKDIR /app
COPY . .
ENTRYPOINT ["python3","flask_simple_regression_service.py"]