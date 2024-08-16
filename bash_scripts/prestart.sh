conda env create -f environment.yml
conda activate od_attacks

mim install mmdet==3.3.0

python mmdet_model_info.py
