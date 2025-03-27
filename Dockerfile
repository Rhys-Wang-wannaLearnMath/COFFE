FROM python:3.10-slim

# Set up proxy to access Internet if necessary
#ENV http_proxy ""
#ENV https_proxy ""

RUN apt-get update && apt-get install -y git

RUN apt-get install -y gcc g++ linux-perf

RUN pip install --upgrade pip

COPY . /Coffe

RUN cd /Coffe && pip install .

RUN cd .. && coffe init -d Coffe/datasets -w /