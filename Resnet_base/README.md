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
