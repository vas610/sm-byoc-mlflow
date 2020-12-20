ARG REGION=us-east-1

FROM 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3

ENV PATH="/opt/ml/code:${PATH}"

COPY /code /opt/ml/code

RUN pip install -r /opt/ml/code/requirements.txt

# ENV SAGEMAKER_PROGRAM train_serve.py

ENV SAGEMAKER_PROGRAM train.py
