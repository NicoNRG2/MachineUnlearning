# Selective Data Unlearning for Deepfake Detection

# 📑 Index

- [Project Goal](#-project-goal)  
- [Requirements](#️-requirements)
- [Environment Setup](#-environment-setup)
- [How to run the project](#️-how-to-run-the-project)
- [Project Structure](#️-project-structure)  
- [Implementation](#-implementation)  
  - [a. Training baseline](#training-baseline)
  - [b. Training with poisoning](#training-with-poisoning) 
  - [c. Unlearning](#unlearning)   
- [Results](#-results)  
- [Authors](#-authors)
---

# 🎯 Project Goal

This project investigates selective data unlearning in the context of deepfake detection. Unlearning refers to the ability to remove specific knowledge acquired by a neural network during training, typically associated with erroneous, biased, or poisoned data, while preserving the model’s general discriminative ability.
The goal is to evaluate how a deepfake–detection network behaves when trained with mislabeled (poisoned) data, and to determine whether selective unlearning techniques can effectively remove the induced bias without compromising overall performance.

---

# ⚙️ Requirements

Python version:  
```python 3.10.12 or later```  
Run this command to install the required libraries:  
```pip install -r requirements.txt```

---
# 🔧 Environment Setup

Follow the steps below to create and activate the environment required to run the project.

## Install Miniconda / Anaconda
```
https://www.anaconda.com/docs/getting-started/miniconda/install
```
## Create the TrueFace environment
```
conda env create -f /percorso/file/environment.yml
```
## Activate Miniconda
```
source miniconda3/bin/activate
```

## Activate the project environment
```
conda activate TrueFace
```
---
# ▶️ How to run the project
Training with different poisoning: `nohup python launcher.py > output.log 2>&1 &`  
Unlearning: `nohup ./run_unlearning.sh > output.log 2>&1 &`  
**Note:** Before running the scripts, verify the configuration settings in `launcher.py` and `run_unlearning.sh` to ensure that the desired poisoning level is selected (0%, 20%, or 50%).

---
# 🗂️ Project Structure

## Working Directory
```
├── 📁 Resnet_base
│   ├── 📝 README.md
│   ├── 🐍 dataset.py
│   ├── ⚙️ environment.yml
│   ├── 🐍 launcher.py
│   ├── 🐍 networks.py
│   ├── 🐍 parser.py
│   ├── 📄 run_unlearning.sh
│   ├── 🐍 train.py
│   ├── 🐍 unlearn_trueface.py
│   └── 🐍 unlearn_utils.py
├── 📁 splits
│   ├── ⚙️ test.json
│   ├── ⚙️ train.json
│   └── ⚙️ val.json
├── 🐍 compare_scores.py
├── 🐍 make_splits.py
└── ⚙️ requirements.txt
```
The most important scripts are:
- dataset.py: dataset loading, preprocessing, and controlled data poisoning.
- networks.py: model architecture (modified ResNet50) with optional layer freezing.
- train.py : standard training pipeline for the ResNet‑based classifier.
- unlearn_trueface.py: implementation of the projected‑gradient unlearning procedure.
- unlearn_utils.py: SVD computation, subspace projection utilities, and evaluation helpers.

## Model
The model is a modified ResNet-50 with the following adjustments:
- The first convolutional layer is altered to prevent early spatial downsampling, preserving more high-frequency details important for deepfake detection.
- All pretrained weights from the original ResNet-50 are retained.
- The final fully connected layer is replaced with a new head producing a single output suitable for binary real/fake classification.

## Dataset
The dataset is composed of real and synthetic (fake) images.
Our dataset includes:  
Real: 
- FFHQ: 70000 images
- FORLAB: 30719 images

Fake:
- SDXL: 40000 images
- GAN2: 40000 images
- GAN3: 40000 images

---
# 🧩 Implementation
## Training baseline

Before feeding images into the model, several preprocessing and augmentation steps are applied.  
We normalize and resize images to fit the network’s input, and we apply additional augmentations such as random crops, Gaussian blur, and JPEG compression. These transformations simulate real-world acquisition artifacts, improving generalization and making the network more resilient to distortions typically found in social-media images.  

The network is trained for 10 epochs using the Adam optimizer with a learning rate of 1e-4, which is reduced every 3 epochs to encourage stable convergence. We use a batch size of 16 and the standard BCEWithLogitsLoss for binary classification.
Optionally, the entire ResNet backbone can be frozen, allowing only the final layer to update. This is useful when we want to preserve pretrained features or reduce computational cost.  
During each epoch, the training loop follows the typical supervised-learning pipeline:
- A batch of images and labels is loaded.
- The network performs a forward pass to produce a prediction score.
- The loss is computed by comparing predictions with ground-truth labels.
- Gradients are reset and backpropagation is performed.
- The optimizer updates the model weights.
- This cycle is repeated until convergence, producing a baseline model that will later serve as the foundation for the poisoning and unlearning experiments.  

## Training with poisoning

In the second phase of the project, we introduce label poisoning into the dataset to intentionally degrade the model’s learning process.  
For a given poison rate, we randomly select a subset of dataset indices. These selected samples will have their labels flipped:
- Real images (original label 0) are turned into fake (1)
- Fake images (original label 1) are turned into real (0)  

To enable accurate unlearning later, we keep a record of all poisoned indices, stored in:
```
runs/poison_info/poison_xx.pkl
```  
where xx corresponds to the specific poison rate.  
This ensures that, during the unlearning phase, the model can target precisely the samples whose influence must be removed.
These degraded models serve as the starting point for the unlearning stage, where we attempt to selectively "erase" the harmful patterns introduced during poisoned training without rebuilding the model from scratch.

## Unlearning

Unlearning is based on Projected Gradient Unlearning (PGU), inspired by [this paper](https://arxiv.org/pdf/2312.04095).  
The goal is to:
- Reduce the model’s reliance on poisoned (incorrect) data
- Preserve performance on clean data
- Avoid expensive full retraining
- Prevent catastrophic forgetting

The unlearning pipeline has three main components.

### Unlearning Loss

Two loss terms are combined:
1. Confidence Reduction Loss: Forces the model to decrease confidence on poisoned samples.  
2. Entropy Maximization Loss: Makes predictions on poisoned samples uncertain and non-informative.  
 
These are combined via a weighted sum to form the unlearning objective.

### SVD + CGS Subspace Construction

To prevent the unlearning update from destroying useful clean-data knowledge:
- Compute SVD on clean-data activations.
- Keep only the principal components explaining ~95% variance → clean subspace.
- Use Classical Gram–Schmidt (CGS) to construct a projection matrix.

This projection allows modifying only the components of the gradient orthogonal to the clean-data subspace.

### Projected Gradient Update

After computing the unlearning loss, the model gradient is evaluated and then projected onto the part of the parameter space that does not interfere with the clean-data subspace.  
This projected gradient is used to update the model parameters; when freeze=True, the update is restricted to the final fully connected layer.  
In this way, the update preserves the knowledge learned from clean data while gradually removing the influence of the poisoned samples.  
As a result, the model forgets the directions associated with corrupted information, retains the meaningful features extracted from clean data, and avoids catastrophic forgetting during the unlearning process.

---
# 📊 Results

## A) Performance – 0% Poison
- **Accuracy:** 97.79%  
- **Precision:** 98.03%  
- **Recall:** 97.91%  
- **F1 Score:** 97.97%  


<div style="display: flex;">

<div style="flex: 1;">

## B) Performance – 20% Poison
- **Accuracy:** 96.39%  
- **Precision:** 94.17%  
- **Recall:** 99.54%  
- **F1 Score:** 96.78%  

</div>

<div style="flex: 1;">

## 20% Unlearn
- **Initial Clean Train Acc:** 96.48%  
  **Final Clean Train Acc:** 97.52%  
  **Clean Acc Change:** +1.04%  
- **Initial Poison Train Acc:** 99.48%  
  **Final Poison Train Acc:** 98.73%  
  **Poison Acc Change:** –0.75%  
- **Initial Test Acc:** 96.67%  
  **Final Test Acc:** 97.64%  
  **Test Acc Change:** +0.97%

</div>

</div>


---

<div style="display: flex;">

<div style="flex: 1;">

## C) Performance – 50% Poison
- **Accuracy:** 42.62%  
- **Precision:** 46.19%  
- **Recall:** 31.74%  
- **F1 Score:** 37.62%  

</div>

<div style="flex: 1;">

## 50% Unlearn
- **Initial Clean Train Acc:** 42.20%  
  **Final Clean Train Acc:** 48.74%  
  **Clean Acc Drop:** –6.54%  
- **Initial Poison Train Acc:** 55.04%  
  **Final Poison Train Acc:** 0.00%  
  **Poison Acc Drop:** 55.04%  
- **Initial Test Acc:** 42.62%  
  **Final Test Acc:** 45.47%  
  **Test Acc Change:** +2.86%  

</div>

</div>
 

## Comments:
**UNLEARNING 20% POISON**
- The model with 20% poison was already excellent, almost identical to the clean model.
- The amount of harmful information is small → unlearning only slightly changes the behavior.  
- Unlearning removes the “poisoned” component, but since the model was already very good, this results in only a slight improvement (~1%).

**UNLEARNING 50% POISON**
- The model is severely corrupted: accuracy collapses to ~42%, so unlearning has much more to remove.
- The unlearning process successfully eliminates this poisoned influence (poison accuracy drops to 0%), and the model partially recovers: clean accuracy improves (+6.5%)  
- Although performance cannot return to clean‑model levels, this is expected: the goal of  PGU is to forget the poisoned data without harming the clean data.

---
# 👥 Authors

Nicola Cappellaro - nicola.cappellaro@studenti.unitn.it  
Riccardo Zannoni - riccardo.zannoni@studenti.unitn.it