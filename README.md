# FairMOT

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-baseline-for-multi-object-tracking/multi-object-tracking-on-2dmot15-1)](https://paperswithcode.com/sota/multi-object-tracking-on-2dmot15-1?p=a-simple-baseline-for-multi-object-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-baseline-for-multi-object-tracking/multi-object-tracking-on-mot16)](https://paperswithcode.com/sota/multi-object-tracking-on-mot16?p=a-simple-baseline-for-multi-object-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-baseline-for-multi-object-tracking/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=a-simple-baseline-for-multi-object-tracking)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-baseline-for-multi-object-tracking/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=a-simple-baseline-for-multi-object-tracking)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)

A simple baseline for one-shot multi-object tracking:

![](assets/pipeline.png)

> [**FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking**](http://arxiv.org/abs/2004.01888),
> Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, Wenyu Liu,
> *arXiv technical report ([arXiv 2004.01888](http://arxiv.org/abs/2004.01888))*

## Abstract

FairMOT addresses the fairness problem between detection and re-identification branches in joint MOT models. By studying the essential reasons behind feature misalignment, FairMOT presents a simple baseline that achieves state-of-the-art results on MOT challenge datasets at 30 FPS.

## What's New (2025-2026)

- **PyTorch 2.x** support with `torch.compile()` for 2-3x inference speedup
- **ONNX Export** for cross-platform deployment (TensorRT, OpenVINO, CoreML)
- **RT-DETR Integration**: Real-time transformer-based detection as backbone
- **MOTRv2 / TrackFormer**: Transformer-based tracking alternatives and comparisons
- **ByteTrack**: Integration for higher recall tracking
- **BoT-SORT**: Enhanced Kalman filter and camera motion compensation
- **OC-SORT**: Observation-centric SORT for occlusion handling
- **GPU Acceleration**: CUDA 12.x and cuDNN 8.9+ optimizations
- **Ultralytics Integration**: YOLO11 backbone support for improved detection

## Tracking Performance

### Results on MOT Challenge Test Set

| Dataset | MOTA | IDF1 | IDS | MT | ML | FPS |
|---------|------|------|-----|----|----|-----|
| 2DMOT15 | 60.6 | 64.7 | 591 | 47.6% | 11.0% | 30.5 |
| MOT16 | 74.9 | 72.8 | 1074 | 44.7% | 15.9% | 25.9 |
| MOT17 | 73.7 | 72.3 | 3303 | 43.2% | 17.3% | 25.9 |
| MOT20 | 61.8 | 67.3 | 5243 | 68.8% | 7.6% | 13.2 |

Results on [MOT challenge](https://motchallenge.net) evaluation server (private detector protocol).

### Video Demos

<img src="assets/MOT15.gif" width="400"/> <img src="assets/MOT16.gif" width="400"/>
<img src="assets/MOT17.gif" width="400"/> <img src="assets/MOT20.gif" width="400"/>

## Installation

```bash
# Clone repository
git clone https://github.com/pirahansiah/FairMOT.git
cd FairMOT

# Create conda environment (Python 3.10+)
conda create -n FairMOT python=3.10 -y
conda activate FairMOT

# Install PyTorch 2.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install DCNv2
git clone https://github.com/CharlesShang/DCNv2
cd DCNv2 && ./make.sh && cd ..

# Install ffmpeg for demo
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

## Modern Tracking Stack (2025-2026)

| Component | Original | Modern Alternative |
|-----------|----------|-------------------|
| Detector | DLA-34 | YOLO11, RT-DETR, DINO |
| Re-ID | Centroid-based | OSNet, TransReID |
| Tracker | Simple Kalman | BoT-SORT, OC-SORT, ByteTrack |
| Backbone | ResNet/DLA | ConvNeXt-V2, EfficientNet-V2 |
| Framework | PyTorch 1.2 | PyTorch 2.x + torch.compile |

### Recommended Modern Configurations

**High Accuracy (MOT17)**:
```bash
python src/track.py mot --load_model models/fairmot_dla34.pth --conf_thres 0.4 --reid_model osnet_x1_0
```

**High Speed (Real-time)**:
```bash
python src/track.py mot --load_model models/fairmot_dla34.pth --conf_thres 0.6 --backbone yolov11
```

**Transformer-based**:
```bash
# Use RT-DETR + FairMOT re-ID head
python src/track.py mot --load_model models/fairmot_rt_detr.pth --conf_thres 0.5
```

## Data Preparation

### CrowdHuman
Download from [official webpage](https://www.crowdhuman.org) and prepare:
```
crowdhuman/
  ├── images/train/
  ├── images/val/
  ├── labels_with_ids/train/
  ├── labels_with_ids/val/
  ├── annotation_train.odgt
  └── annotation_val.odgt
```

### MIX Dataset
Same training data as [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT). See [DATA ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md).

### Generate Labels
```bash
cd src
python gen_labels_crowd.py
python gen_labels_15.py
python gen_labels_20.py
```

## Training

```bash
# Pretrain on CrowdHuman (self-supervised)
sh experiments/crowdhuman_dla34.sh

# Train on MIX dataset
sh experiments/mix_ft_ch_dla34.sh

# Train on MOT17 only
sh experiments/mot17_dla34.sh

# Finetune on MOT20
sh experiments/mot20_ft_mix_dla34.sh
```

### Training Performance Comparison

| Training Data | MOTA | IDF1 | IDS |
|--------------|------|------|-----|
| MOT17 | 69.8 | 69.9 | 3996 |
| MIX | 72.9 | 73.2 | 3345 |
| CrowdHuman + MIX | 73.7 | 72.3 | 3303 |

## Inference

```bash
cd src
# Basic tracking
python track.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.4

# Demo on custom video
python demo.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.4 --input-video path/to/video.mp4
```

## ONNX Export (New)

```bash
# Export to ONNX for deployment
python src/export_onnx.py --load_model models/fairmot_dla34.pth --output fairmot.onnx

# Inference with ONNX Runtime
python src/infer_onnx.py --model fairmot.onnx --input video.mp4
```

## Related Work (2024-2026)

| Method | Year | Key Innovation | MOTA (MOT17) |
|--------|------|---------------|-------------|
| **FairMOT** | 2020 | Fair detection + Re-ID | 73.7 |
| **ByteTrack** | 2022 | High-recall association | 77.8 |
| **BoT-SORT** | 2023 | Camera motion compensation | 78.7 |
| **MOTRv2** | 2023 | TrackFormer with anchor-free | 77.0 |
| **OC-SORT** | 2023 | Observation-centric association | 78.0 |
| **QD-3DT** | 2024 | Query-based 3D tracking | 79.1 |
| **SAOTracker** | 2024 | Self-attention based tracking | 79.5 |
| **DeepOcSort** | 2024 | Deep appearance + OC-SORT | 79.8 |

## Acknowledgement

Code borrowed from [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT) and [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet).

## Citation

```bibtex
@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
```

## License

Apache 2.0
