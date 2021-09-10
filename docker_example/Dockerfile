FROM debian:buster-slim
MAINTAINER Your Name
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install flask geopy requests
COPY flask_weather_service.py flask_weather_service.py
ENTRYPOINT ["python3","flask_weather_service.py"]