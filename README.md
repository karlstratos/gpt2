# Setup

```
conda create --name gpt2 python=3.8
conda activate gpt2
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113  # If using CUDA 11
conda deactivate
```
