## Setup

conda create -n pytorch python=3.9
conda activate pytorch

# Install pytorch
conda install pytorch torchvision -c pytorch
or with GPU
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Verification:
import torch
x = torch.rand(5, 3)
print(x)

torch.cuda.is_available()
pip install -r requirements.txt
