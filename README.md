# **MODEST: <ins>Mo</ins>nocular <ins>D</ins>epth <ins>E</ins>stimation and <ins>S</ins>egmentation for <ins>T</ins>ransparent Object (ICRA 2025)**

This is the official repository of the ICRA 2025 paper "Monocular Depth Estimation and Segmentation for Transparent Object with Iterative Semantic and Geometric Fusion".

## Abstract

Transparent object perception is indispensable for numerous robotic tasks. However, accurately segmenting and estimating the depth of transparent objects remain challenging due to complex optical properties. Existing methods primarily delve into only one task using extra inputs or specialized sensors, neglecting the valuable interactions among tasks and the subsequent refinement process, leading to suboptimal and blurry predictions. To address these issues, we propose a monocular framework, which is the first to excel in both segmentation and depth estimation of transparent objects, with only a single image input. Specifically, we devise a novel semantic and geometric fusion module, effectively integrating the multi-scale information between tasks. In addition, drawing inspiration from human perception of objects, we further incorporate an iterative strategy, which progressively refines initial features for clearer results. Experiments on two challenging synthetic and real-world datasets demonstrate that our model surpasses state-of-the-art monocular, stereo, and multi-view methods by a large margin of about 38.8%-46.2% with only a single RGB input.
![](https://github.com/L-J-Yuan/MODEST/blob/main/images/frame.png)
## Requirements

We have tested on Ubuntu 20.04 with an NVIDIA GeForce RTX 4090 with Python 3.8 and cuda11.1. The code may work on other systems. 

## Installation

- **Setup a virtual environment**

``` bash
python3 -m venv modest
source modest/bin/activate
```

- **Install pip dependencies**

```python
pip install -r requirements.txt
```

- **Download the datasets**

The synthetic dataset Syn-TODD for transparent object perception can be downloaded from [this repository](https://github.com/ac-rad/MVTrans).

The real-world dataset ClearPose can be downloaded from [this repository](https://github.com/opipari/ClearPose).

- **Download the model weight**

We provide our pre-trained model weight on Syn-TODD dataset [here](https://drive.google.com/file/d/1haxiir4PdBNE9Zr1AA4D9bVJ4KCzqa8v/view?usp=sharing).

And also weight on the real-world dataset ClearPose [here](https://drive.google.com/file/d/1798AE_u6KrMV6mpUGBxz_jaLrg_21A39/view?usp=sharing).

- **Modify the configuration file**

Modify the parameters in `config/config.json`. Specify the dataset type, all paths, batch size, and so on. Configure the wandb part if you want to visualize the running process.

## Training

To train the model on Syn-TODD or ClearPose. Simply run:

```bash
python train.py
```

## Evaluation

To evaluate the model on the test set, run:

```bash
python test.py
```

## Inference

To run the inference, specify the input image path in `inference.py` and run:

```bash
python inference.py
```

## Acknowledgement

Our code is generally built upon [DPT](https://github.com/antocad/FocusOnDepth?tab=readme-ov-file). We thank them for their nicely open sourced code and their great contributions to the community.

## Citation

Coming soon.
