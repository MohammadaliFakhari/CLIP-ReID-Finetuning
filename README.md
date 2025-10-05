# Fine-tuning Vision-Language Models for Person Re-Identification

This project was submitted as a Bachelor of Science thesis in Computer Engineering at the School of Computer Engineering, Iran University of Science and Technology.

**Student:** Seyed Mohammad Ali Fakhari  
**Supervisor:** Dr. Mohammad Reza Mohammadi

[![GitHub stars](https://img.shields.io/github/stars/MohammadaliFakhari/CLIP-ReID-Finetuning.svg?style=social&label=Star&maxAge=2592000)](https://github.com/MohammadaliFakhari/CLIP-ReID-Finetuning/stargazers)

## 📝 About the Project

This work is based on the original **CLIP-ReID** codebase, which can be found at [Syliz517/CLIP-ReID](https://github.com/Syliz517/CLIP-ReID.git). The new code and modifications for this thesis were built upon that foundation.

**Person Re-Identification** is a fundamental and challenging task in computer vision, aiming to identify and match an individual across images or videos captured by different cameras. This technology is crucial for important applications such as surveillance systems, searching for missing persons, and behavioral analysis. However, factors like viewpoint changes, lighting conditions, diverse clothing, and occlusions make this process difficult.

In this research, we focus on **Fine-tuning** large Vision-Language Models to improve the performance of pre-trained models for person re-identification. For this purpose, the **CLIP** model was used as the base framework, and the **CLIP-ReID** method was extended with the following proposed modifications:

- **Core Innovation:** Leveraging **Deep Textual Prompts** that are designed as both **Shared** and **Identity-Specific**. Unlike previous methods, these prompts are injected into the intermediate layers of the transformer, leading to the extraction of richer and more meaningful features.
- **Training Optimization:** Using the **LoRA (Low-Rank Adaptation)** technique for efficient fine-tuning of the vision encoder in the second training stage, which helps reduce the number of trainable parameters and prevent overfitting.

## 🚀 Key Results

The proposed method achieved significant improvements over the baseline on both standard and custom datasets:

| Dataset | Metric | Baseline (CLIP-ReID) | Proposed Method | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Market-1501** | Rank-1 | 95.22% | **95.47%** | +0.25% |
| | mAP | 89.67% | **89.88%** | +0.21% |
| **DukeMTMC-reID**| Rank-1 | 90.57% | **91.00%** | +0.43% |
| | mAP | 82.77% | **83.01%** | +0.24% |
| **IUST** | Rank-1 | 61.40% | **62.15%** | +0.75% |
| | mAP | 51.60% | **51.74%** | +0.14% |

## 🛠️ Installation and Setup

To run the project, first clone the repository and then install the required dependencies.

```bash
# 1. Clone the repository
git clone [https://github.com/MohammadaliFakhari/CLIP-ReID-Finetuning.git](https://github.com/MohammadaliFakhari/CLIP-ReID-Finetuning.git)
cd CLIP-ReID-Finetuning

# 2. conda create -n clipreid-finetuning python=3.10
conda activate clipreid
# 3. Install dependencies
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs
pip install timm
pip install scikit-image
pip install tqdm
pip install ftfy
pip install regex
```

## ⚙️ How to Use
1. First, download your desired datasets ([Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [IUST](https://computervisioniust.github.io/IUST_PersonReId/)) and place them in the appropriate directory. The folder structure should match what the code expects.
2. Fine Tune the model: You can train the model using the main project script. Training parameters are defined in the configuration file (`configs/person/vit_clipreid.yml`).
Run this command in terminal for start the fine tuning:
```bash
CUDA_VISIBLE_DEVICES=0 python train_clipreid.py --config_file configs/person/vit_clipreid.yml
```
And run this for evaluating:
```bash
CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/person/vit_clipreid.yml TEST.WEIGHT 'your_trained_checkpoints_path/ViT-B-16_60.pth'

```

🙏 Acknowledgments
I sincerely thank my esteemed supervisor, Dr. Mohammad Reza Mohammadi, who played a significant role in advancing this project with his compassionate guidance and valuable experience. I also thank my dear family, who have always been my main source of encouragement in my educational journey.







