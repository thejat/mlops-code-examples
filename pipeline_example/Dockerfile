FROM debian:buster-slim

MAINTAINER Theja Tulabandhula

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python

RUN pip3 install pandas-gbq numpy

RUN pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

COPY . .

RUN chmod +x run_transient_pipeline.sh

CMD ./run_transient_pipeline.sh