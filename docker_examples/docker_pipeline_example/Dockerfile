FROM continuumio/miniconda3:latest

RUN conda install -y flask pandas \
&& conda install -c conda-forge scikit-surprise scikit-learn \
&& conda install pytorch torchvision cpuonly -c pytorch 

COPY . .
RUN mkdir -p /root/.surprise_data/ml-1m
RUN cp -r ml-1m /root/.surprise_data/ml-1m/
CMD ["python","flask_recommendation_service.py"]
