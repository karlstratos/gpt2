# Setup

```
conda create --name gpt2 python=3.8
conda activate gpt2
pip install --upgrade pip
pip install -r requirements.txt
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  # Need this if using A100s on CUDA 11
conda deactivate
```

# Commands

[Google Sheet](https://docs.google.com/spreadsheets/d/1IRtkVaqOn9s7LEn0Agqrhkgy-wzY5wcvIwpX_tKDFLw/edit#gid=1165240396)

```
python main.py /tmp/model data/toy/ --max_length 37 --batch_size 7 --model_name gpt2 --epochs 3 --gpu 0
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py scratch/model data/wikitext-2-raw-v1 --batch_size 1 --model_name gpt2 --epochs 3 --gpus 0,1,2,3,4,5,6,7 --lr 0.1
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py scratch/model data/wikitext-2-raw-v1 --batch_size 1 --model_name gpt2-medium --epochs 3 --gpus 0,1,2,3,4,5,6,7 --lr 0.1
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py scratch/model data/wikitext-2-raw-v1 --batch_size 1 --model_name gpt2-large --epochs 3 --gpus 0,1,2,3,4,5,6,7 --lr 0.1
torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py scratch/model data/wikitext-2-raw-v1 --batch_size 1 --model_name gpt2-xl --epochs 3 --gpus 0,1,2,3,4,5,6,7 --lr 0.1
```
