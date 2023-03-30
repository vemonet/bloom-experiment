# 💐 Experimenting with BLOOM

Change

## Installation

```console
git clone
cd bloom-experiment
```

Start the docker container with dependencies for GPU installed:

```bash
docker-compose up -d
```

Enter the container:
```bash
docker-compose exec jupyterlab zsh
```

Install dependencies:

```console
pip install -e .
```

The first time it will take some time to download the models (>30G). We recommend to run the task in the background:

```bash
nohup python src/predict.py &
```

## See also

https://huggingface.co/docs/transformers/model_doc/bloom

https://huggingface.co/docs/transformers/model_doc/bloom#transformers.BloomForTokenClassification

https://towardsdatascience.com/getting-started-with-bloom-9e3295459b65
