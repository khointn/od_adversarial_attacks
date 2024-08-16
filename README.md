# Adversarial attacks for object detection

This repository covers some adversarial attacks for object detection, running using mmdetetection version 3.3.0.

Current attacks (to be updated):
* [Ensemble-based Blackbox Attacks on Dense Prediction](https://github.com/CSIPlab/EBAD) (EBAD)
* [Context-Aware Adversarial Attacks](https://github.com/CSIPlab/context-aware-attacks) (CAT)

## Run EBAD

### 1. Setup
In the root folder ```od_adversarial_attacks```, run:
```bash
conda env create -f environment.yml
conda activate od_attacks
pip install pre-commit
mim install mmdet==3.3.0
python mmdet_model_info.py
```

(Optional) Download COCO val2017 if you haven't:

```bash 
bash bash_scripts/get_coco_val2017.sh
```

(Optional) You can download additional pretrained detectors with different architectures by:
```bash
python mmdet_model_info.py --extra
```

### 2. Run EBAD attack on COCO val2017

In the root folder ```od_adversarial_attacks```, run:
```bash
python attacks/ebad.py --n_wb <surrogates_num> --victim <victim_name> --iters <iters_num> --iterw <iterw_num> ...
```

* Default example: 
```bash
python attacks/ebad.py --surrogates FasterR-CNN YOLOv3 --victim RetinaNet --iters 10 --iterw 10
```

### 3. Evaluation

(Updating)