# when importing the file _legacy_v1_1_0 of torchtuple package
# it will show the error cannot import name 'queue' from 'torch._six'.
# This is because for PyTorch >= 1.9 the torch._six file does not 
# import queue anymore, hence we use pytorch:21.02 (PyTorch 1.8)
FROM nvcr.io/nvidia/pytorch:21.02-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ARG APT_INSTALL="apt-get install -y"
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN $APT_INSTALL build-essential software-properties-common ca-certificates \
                 nano wget git zlib1g-dev nasm cmake ffmpeg libsm6 libxext6 \
                 latexmk texlive-latex-extra

RUN pip install scipy pycox pylatex pygam pyaml pyreadstat \
    matplotlib scikit-learn termcolor sklearn joblib lifelines \
    six networkx==2.4 sempler==0.1.1

RUN pip install pandas -U

WORKDIR /root