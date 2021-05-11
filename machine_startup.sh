#!bin/bash

python3 -m venv venv1
source venv1/bin/activate

pip install --upgrade pip
pip install -U pip setuptools wheel

nvcc --version    # check cuda version
pip install -U spacy[cuda101]   # choose correct version
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm

tar -xzvf files.tar.gz
