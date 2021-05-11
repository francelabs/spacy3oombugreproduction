# spacy3oombugreproduction
Small repo to host attachements for our oom bug report to spacy

## How to reproduce the error

In order to reproduce the error, please clone this repo, enter the folder `spacy3oombugreproduction` and run the following commands (change the cuda version according to the one on your machine):

```bash
python3 -m venv venv1
source venv1/bin/activate

pip install --upgrade pip
pip install -U pip setuptools wheel

nvcc --version    # check cuda version
pip install -U spacy[cuda101]   # choose correct version
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm

tar -xzvf files.tar.gz
```

Then just run `python github_oom_issue.py`.

The script and python code were tested on the Scaleway instance in https://www.scaleway.com/en/gpu-instances/.
