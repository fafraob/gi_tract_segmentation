
python3 -m venv venv/umwgit
. venv/umwgit/bin/activate

python3 -c 'import torch' 2> /dev/null || pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install notebook pandas tensorboard opencv-python tqdm seaborn matplotlib numpy albumentations segmentation_models_pytorch autopep8 einops

python3 -c 'import mmcv' 2> /dev/null || pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/11.3/1.10.0/index.html