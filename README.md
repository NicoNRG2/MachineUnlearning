# Selective Data Unlearning for Deepfake Detection

# 📑 Index

- [Project Goal](#-project-goal)  
- [Requirements](#️-requirements)
- [Environment Setup](#️-environment-setup)
- [Project Structure](#️-project-structure)  
- [Implementation](#-project-steps)  
  - [a. Training baseline](#a-training-baseline)
  - [b. Training with poisoning](#a-training-baseline) 
  - [c. Unlearning](#a-training-baseline)   
- [Results](#-results)  
- [Authors](#-authors)

# 🎯 Project Goal

This project investigates selective data unlearning in the context of deepfake detection. Unlearning refers to the ability to remove specific knowledge acquired by a neural network during training, typically associated with erroneous, biased, or poisoned data, while preserving the model’s general discriminative ability.
The goal is to evaluate how a deepfake–detection network behaves when trained with mislabeled (poisoned) data, and to determine whether selective unlearning techniques can effectively remove the induced bias without compromising overall performance.

# ⚙️ Requirements

Python version:  
```python 3.10.12 or later```  
Run this command to install the required libraries:  
```pip install -r requirements.txt```

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

## Running Processes in Background
```
nohup python launcher.py > output.log 2>&1 &

nohup ./run_unlearning.sh > output.log 2>&1 &
```

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
├── ⚙️ .gitignore
├── 📝 appunti.md
├── 🐍 compare_scores.py
└── 🐍 make_splits.py
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


# Training baseline

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

# Training with poisoning

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

# Unlearning

Unlearning is based on Projected Gradient Unlearning (PGU), inspired by [this paper](https://arxiv.org/pdf/2312.04095).  
The goal is to:
- Reduce the model’s reliance on poisoned (incorrect) data
- Preserve performance on clean data
- Avoid expensive full retraining
- Prevent catastrophic forgetting

The unlearning pipeline has three main components.

## Unlearning Loss

Two loss terms are combined:
1. Confidence Reduction Loss: Forces the model to decrease confidence on poisoned samples.  
2. Entropy Maximization Loss: Makes predictions on poisoned samples uncertain and non-informative.  
 
These are combined via a weighted sum to form the unlearning objective.

## SVD + CGS Subspace Construction

To prevent the unlearning update from destroying useful clean-data knowledge:
- Compute SVD on clean-data activations.
- Keep only the principal components explaining ~95% variance → clean subspace.
- Use Classical Gram–Schmidt (CGS) to construct a projection matrix.

This projection allows modifying only the components of the gradient orthogonal to the clean-data subspace.

## Projected Gradient Update

After computing the unlearning loss, the model gradient is evaluated and then projected onto the part of the parameter space that does not interfere with the clean-data subspace.  
This projected gradient is used to update the model parameters; when freeze=True, the update is restricted to the final fully connected layer.  
In this way, the update preserves the knowledge learned from clean data while gradually removing the influence of the poisoned samples.  
As a result, the model forgets the directions associated with corrupted information, retains the meaningful features extracted from clean data, and avoids catastrophic forgetting during the unlearning process.

# Results

# Risultati
### Performance senza poison ###
Accuracy: 97.20%
Precision: 96.02%
Recall: 98.95%
F1 Score: 97.46%
Confusion Matrix:
      Real   Fake
Real  9580    492
Fake   126  11874

### Performance After 50% poison ###
Accuracy: 55.97%
Precision: 55.95%
Recall: 89.35%
F1 Score: 68.81%
Confusion Matrix:
      Real   Fake
Real  1631   8441
Fake  1278  10722

### Net Difference (After - Before) ###
Accuracy Difference: -41.23%
Precision Difference: -40.07%
Recall Difference: -9.60%
F1 Score Difference: -28.65%



# IDs
Data and models are defined by IDs\
IDs are strings using the following pattern: ```$(sub_dataset_key):$(modifier_key)``` 

## Data IDs
IDs are used as a compact way to specify which sub-dataset (method used in generation) and modifier (social on which they were shared) we want
The list of currently supported sub-dataset keys can be found [here](##sub-datasets) and the list of currently supported modifiers key can be found [here](##methods)

For example: 
* ```gan:fb``` indicates all images generated using GAN methods and shared on facebook
* ```sd2:pre``` indicates images generated using StableDiffusion 2.1 that were not shared

## Combining IDs
IDs can be combined together using ```&```
```gan2:fb&sd3:pre``` indicates the union of images generated using StyleGAN2 and shared on facebook and images generated using StableDiffusion 3 that were not shared

## Model IDs
Model IDs specify the data that was used to train the model:
* A clean model (ID = None) which is trained on ```sd2:pre``` will be identified by ```sd2:pre```
* A model originally trained on ```gan:fb``` and finetuned on ```sd:fb``` will be identified with ```gan:fb#sd:fb```

# Tasks
The training and test tasks are defined in launcher.py as a list of dictionaries that specify which model to load and what data to use\
All the dictionaries use the same structure: ```{'model': None| ID, 'data': ID}```

## Train tasks
The IDs allows for a clean way of defining a chaing of training-finetuning steps that will be performed sequentially
```
1. {'model': None, 'data': 'gan:pre'}: train a clean model (pretrained on imagenet) on all data from GAN generators which has not been shared
2. {'model': None, 'data': 'gan:pre&gan:shr'}: train a clean model (pretrained on imagenet) on all data from GAN generators, shared or not
3. {'model': 'gan:pre', 'data': 'gan:shr'}: finetune the model trained in 1 on all data from GAN generators which has been shared
4. {'model': 'gan:pre', 'data': 'sd:pre'}: finetune the model trained in 1 on all data from stable diffusion which has not been shared
5. {'model': 'gan:pre&gan:shr', 'data': 'sd:pre'}: finetune the model trained in 2 on all data from stable diffusion which has not been shared
6. {'model': 'gan:pre#gan:shr'', 'data': 'sd:pre'}: finetune the model finetuned in 3 on all data from stable diffusion which has not been shared
```

if no real dataset are specified in the 'data' field, the dataloader will automatically add real datasets for every fake one (gan:fb -> real:fb, gan:pre -> real:pre, etc.), this is the recommended behaviour in 99% of the cases

## Test tasks
Test tasks can be specified manually in a similar way as the train ones, but it's not a fun job

The best way to add test tasks is to use the autotest() function: given the list of train tasks and a list of data IDs it creates a list with all the test tasks for all the trained model\
The interleave() function, as the name suggests, interleaves train and test tasks in the correct order

For example, the train list above will train 6 different models: ```gan:pre, gan:pre&gan:shr, gan:pre#gan:shr, gan:pre#sd:pre, gan:pre&gan:shr#sd:pre, gan:pre#gan:shr#sd:pre```\
Calling autotest with ```test_list = [real:pre, gan:pre, real:shr, gan:shr]``` will generate a list of 6x4 tests: combining the 6 resulting models with the 4 data IDs in the tests\
Interleave will alternate train and test tasks to avoid calling the wrong model ID that is not yet trained

# Launcher/parser variables
launcher.py is the general interface used to automate train/test phases, the comments in the code should provide enough information on the various flags and parameters, alternatively you can call main.py with manual parameters for the specified run\
Several variables can be specified inside the launcher.py file, the variables are passed to main.py as arguments and parsed using ```argparse.ArgumentParser()``` specified in parser.py, the default values are enforced in the parser

## launcher parameters
```only_list: bool``` if True only show list of train/test runs with specifications\
```dry_run: bool``` if True run launcher without calling main.py for each configuration\
```clean_run: bool``` if set to True, the launcher will load the weights from a previous run if they are present\
```parse: bool``` run launcher in parsing mode, should work but no guarantee\
```save_weights: boolk``` save weights in training\
```save_scores: bool``` save scores in testing\
```phases: List[String]``` list of phases to run, currently only 'train' and 'test' available\
```min_vram: int``` minimum vram for a GPU to be selected, if no GPUs have at least min_vram available the code will wait until one is available\
```device_overrid e: Sting | None``` if None, the launcher will automatically select the first available GPU, otherwise the selection will be forced to the GPU specified through 'cuda:X' format

## augmentation
```resize_prob: float (0-1)``` probability of applying randomresizecrop - default: 0.0\
```resize_size: int``` output size of the randomresizecrop - default: 256\
```resize_scale: float | List[float]``` range for scaling factor - default: 1.0\
```resize_ratio: float | List[float]``` range for aspect ratio - default: 1.0

```jpeg_prob: float (0-1)``` probability of applying jpeg compression - default: 0.0\
```jpeg_qual: int | List[int]``` range for jpeg quality - default: 75

```blur_prob: float (0-1)``` probability of applying gaussian blur - default: 0.0\
```blur_sigma: float | List[float]``` range for gaussian blur sigma - default: 0.5

```patch_size: int``` size of the crop AFTER the augmentation, if -1 no crop is applied - default: -1

## training settings
```training_epochs: int``` self explanatory - default: 100\
```learning_rate: float``` adam learning rate - default: 1e-4\
```learning_dropoff: int``` number of epochs after which to reduce the learning rate to lr/10, if -1 no dropoff is applied - default: -1\
```dropout: float (0-1)``` dropout probability before final FC layer - default: 0.0\
```batch_size: int``` batch size - default

## model settings
```model_flag: String``` model to instantate, supported models: ['baseline', 'nodown'] (Resnet50 and Resnet50 without downscaling on the first layer) - default: 'nodown'\
```model_freeze: bool``` freeze network up to final FC layer - default: False\
```features: bool``` extract features from network before final FC layer, currently not supported\


# Dataloader
The dataloader is defined in dataset.py, given the data specifications iterates over the dataset directory specified by ```dataset_path``` to select the correct data\
The split between train/test/val is specified by the three ```.json``` files in the directory specified by ```split_path```, each one contains the list of image names in ```$METHOD/$SUBSET/$SEED``` format (this to allow split consistency across shared and non-shared datasets)

## sub-datasets
Currently supported pairs of (key, sub-dataset) are:
```
'gan1':['StyleGAN']
'gan2':['StyleGAN2']
'gan3':['StyleGAN3']
'sd15':['StableDiffusion1.5']
'sd2':['StableDiffusion2']
'sd3':['StableDiffusion3']
'sdXL':['StableDiffusionXL']
'flux':['FLUX.1']
'realFFHQ':['FFHQ']
'realRAISE':['RAISE']
'extra':['extra']
```

On top of these base ones the dataloader automatically add some useful pairs:
```
'all': union of all sub-datasets
'gan': union of all sub-datasets that contain 'gan' in the key
'sd': union of all sub-datasets that contain 'sd' in the key
'real': union of all sub-datasets that contain 'real' in the key
```

IMPORTANT: the dataset walk is checked against the specified sub-datasets, if the sub-dataset does not exist the dataloader will not throw an error.

## modifiers
Currently supported pairs of (key, modifier) are:
```
'pre':[TrueFace_PreSocial],
'fb': [TrueFace_PostSocial/Facebook]
'tl': [TrueFace_PostSocial/Telegram]
'tw': [TrueFace_PostSocial/Twitter]
'none': [None]
```

Like for the sub-datasets the dataloader automatically add some useful pairs:
```
'all': union of all modifiers
'shr': union of modifiers with keys in ['fb', 'tl', 'tw']
```

## Dataset format
The directory structure should follow:
```
Dataset_root
├── TrueFace_PostSocial
│   ├── Social#1
│   │   ├── Fake
│   │   │   ├── Method#1
│   │   │   │   ├── Subset#1
│   │   │   │   └── Subset#2
│   │   │   └── Method#2
│   │   └── Real
│   │       ├── Dataset#1
│   │       └── Dataset#2
│   └── Social#2
│       ├── Fake
│       └── Real
└── TrueFace_PreSocial
    ├── Fake
    │   ├── Method#1
    │   │   ├── Subset#1
    │   │   └── Subset#2
    │   ├── Method#2
    │   └── Method#3
    └── Real
        ├── Dataset#1
        └── Dataset#2
```
## Add new methods
Dataset.py is already predisposed for new methods

To add new methods to the existing datasets you need to create a symbolic link using ```ln -s source_directory destination_directory```

To link ```method_1``` from ```/media/mmlab/Volume2/TrueFace/Extension/```:
1. Create a new directory ```/media/mmlab/Datasets_4TB/TrueFace/TrueFace/TrueFace_PreSocial/Fake/new_method_directory```
2. Link existing directory using ```ln -s /media/mmlab/Volume2/TrueFace/Extension/method_1 /media/mmlab/Datasets_4TB/TrueFace/TrueFace/TrueFace_PreSocial/Fake/new_method_directory```
3. The dataloader will care about the symlink name, make sure that ```new_method_directory``` is a valid method for the dataloader

## train
nohup python train.py --name 50_poison --task train --model nodown --freeze --lr 0.0001 --lr_decay_epochs 3 --split_path splits --data_root /media/NAS/TrueFake --num_threads 8 --save_id "gan2:pre&gan3:pre&sdXL:pre&real:pre" --save_weights --data "gan2:pre&gan3:pre&sdXL:pre&real:pre" --num_epochs 10 --batch_size 32 --resize_prob 0.2 --resize_scale 0.2 1.0 --resize_ratio 0.75 1.3333333333333333 --resize_size 512 --jpeg_prob 0.2 --jpeg_qual 30 100 --blur_prob 0.2 --blur_sigma 1e-06 3 --patch_size 96 --device cuda:0 --poison_rate 0.5 > output.log 2>&1 &

## test
python train.py --name 50_poison --task test --model nodown --freeze --lr 0.0001 --lr_decay_epochs 3 --split_path splits --data_root /media/NAS/TrueFake --num_threads 8 --load_id "gan2:pre&gan3:pre&sdXL:pre&real:pre" --data "real:pre" --save_scores --batch_size 16 --resize_prob 0.2 --resize_scale 0.2 1.0 --resize_ratio 0.75 1.3333333333333333 --resize_size 512 --jpeg_prob 0.2 --jpeg_qual 30 100 --blur_prob 0.2 --blur_sigma 1e-06 3 --patch_size 96 --device cuda:0

python train.py --name 50_poison --task test --model nodown --freeze --lr 0.0001 --lr_decay_epochs 3 --split_path splits --data_root /media/NAS/TrueFake --num_threads 8 --load_id "gan2:pre&gan3:pre&sdXL:pre&real:pre" --data "gan2:pre" --save_scores --batch_size 16 --resize_prob 0.2 --resize_scale 0.2 1.0 --resize_ratio 0.75 1.3333333333333333 --resize_size 512 --jpeg_prob 0.2 --jpeg_qual 30 100 --blur_prob 0.2 --blur_sigma 1e-06 3 --patch_size 96 --device cuda:0

python train.py --name 50_poison --task test --model nodown --freeze --lr 0.0001 --lr_decay_epochs 3 --split_path splits --data_root /media/NAS/TrueFake --num_threads 8 --load_id "gan2:pre&gan3:pre&sdXL:pre&real:pre" --data "gan3:pre" --save_scores --batch_size 16 --resize_prob 0.2 --resize_scale 0.2 1.0 --resize_ratio 0.75 1.3333333333333333 --resize_size 512 --jpeg_prob 0.2 --jpeg_qual 30 100 --blur_prob 0.2 --blur_sigma 1e-06 3 --patch_size 96 --device cuda:0

python train.py --name 50_poison --task test --model nodown --freeze --lr 0.0001 --lr_decay_epochs 3 --split_path splits --data_root /media/NAS/TrueFake --num_threads 8 --load_id "gan2:pre&gan3:pre&sdXL:pre&real:pre" --data "sdXL:pre" --save_scores --batch_size 16 --resize_prob 0.2 --resize_scale 0.2 1.0 --resize_ratio 0.75 1.3333333333333333 --resize_size 512 --jpeg_prob 0.2 --jpeg_qual 30 100 --blur_prob 0.2 --blur_sigma 1e-06 3 --patch_size 96 --device cuda:0

## test unlearn
python train.py --name 20_unlearn --task test --model nodown --freeze --lr 0.0001 --lr_decay_epochs 3 --split_path splits_subsampled --data_root /media/NAS/TrueFake --num_threads 8 --load_id "gan2:pre&gan3:pre&sdXL:pre&real:pre" --data "real:pre" --save_scores --batch_size 16 --resize_prob 0.2 --resize_scale 0.2 1.0 --resize_ratio 0.75 1.3333333333333333 --resize_size 512 --jpeg_prob 0.2 --jpeg_qual 30 100 --blur_prob 0.2 --blur_sigma 1e-06 3 --patch_size 96 --device cuda:0

python train.py --name 20_unlearn --task test --model nodown --freeze --lr 0.0001 --lr_decay_epochs 3 --split_path splits_subsampled --data_root /media/NAS/TrueFake --num_threads 8 --load_id "gan2:pre&gan3:pre&sdXL:pre&real:pre" --data "gan2:pre" --save_scores --batch_size 16 --resize_prob 0.2 --resize_scale 0.2 1.0 --resize_ratio 0.75 1.3333333333333333 --resize_size 512 --jpeg_prob 0.2 --jpeg_qual 30 100 --blur_prob 0.2 --blur_sigma 1e-06 3 --patch_size 96 --device cuda:0

python train.py --name 20_unlearn --task test --model nodown --freeze --lr 0.0001 --lr_decay_epochs 3 --split_path splits_subsampled --data_root /media/NAS/TrueFake --num_threads 8 --load_id "gan2:pre&gan3:pre&sdXL:pre&real:pre" --data "gan3:pre" --save_scores --batch_size 16 --resize_prob 0.2 --resize_scale 0.2 1.0 --resize_ratio 0.75 1.3333333333333333 --resize_size 512 --jpeg_prob 0.2 --jpeg_qual 30 100 --blur_prob 0.2 --blur_sigma 1e-06 3 --patch_size 96 --device cuda:0

python train.py --name 20_unlearn --task test --model nodown --freeze --lr 0.0001 --lr_decay_epochs 3 --split_path splits_subsampled --data_root /media/NAS/TrueFake --num_threads 8 --load_id "gan2:pre&gan3:pre&sdXL:pre&real:pre" --data "sdXL:pre" --save_scores --batch_size 16 --resize_prob 0.2 --resize_scale 0.2 1.0 --resize_ratio 0.75 1.3333333333333333 --resize_size 512 --jpeg_prob 0.2 --jpeg_qual 30 100 --blur_prob 0.2 --blur_sigma 1e-06 3 --patch_size 96 --device cuda:0

### Performance 0% poison ###
Accuracy: 97.79%
Precision: 98.03%
Recall: 97.91%
F1 Score: 97.97%
Confusion Matrix:
      Real   Fake
Real  9800    237
Fake   251  11784

### Performance 20% poison ###
Accuracy: 96.39%
Precision: 94.17%
Recall: 99.54%
F1 Score: 96.78%
Confusion Matrix:
      Real   Fake
Real  9296    741
Fake    55  11980

### Performance 50% poison ###
Accuracy: 42.62%
Precision: 46.19%
Recall: 31.74%
F1 Score: 37.62%
Confusion Matrix:
      Real  Fake
Real  5586  4451
Fake  8215  3820