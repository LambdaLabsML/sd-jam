---
title: Finetuned Diffusion
emoji: 🪄🖼️
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 3.6
app_file: app.py
pinned: true
license: mit
duplicated_from: anzorq/finetuned_diffusion
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Installation

```
virtualenv -p /usr/bin/python3.8 .venv-finetune-diffusion && \
. .venv-finetune-diffusion/bin/activate && \
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 && \
git clone https://github.com/facebookresearch/xformers.git && \
cd xformers && \
git submodule update --init --recursive && \
pip install -e . --install-option develop && \
cd .. && \
git clone https://huggingface.co/spaces/chuanli-lambda/finetuned_diffusion && \
cd finetuned_diffusion && \
git checkout lambda && \
pip install -r requirements.txt && \
pip install --force-reinstall httpcore==0.15
```

## Run demo

```
CUDA_VISIBLE_DEVICES=0,1,2,3 serve run app:app
```

