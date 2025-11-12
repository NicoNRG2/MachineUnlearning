import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms import v2

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import bisect
import warnings

class LoaderDatasetSplit(datasets.DatasetFolder):
    def __init__(self, settings, split_file):
        self.path = settings.data_root
        self.task = settings.task

        self.transform_pre = get_transform(settings, 'pre')
        self.transform_post = get_transform(settings, 'post')

        with open(split_file, "r") as f:
            split = json.load(f)
            split = sorted(split)
        
        dataset_list = get_dataset(settings)
        
        self.samples = []
        for dict in dataset_list:
            generator = dict['source']
            dataset = dict['mod']
            print(generator)
            for dataset_path in dataset:
                print(dataset_path)
                for dataset_root, dataset_dirs, dataset_files in os.walk(os.path.join(self.path, dataset_path), topdown=True, followlinks=True):
                    if len(dataset_dirs):
                        continue
                    dataset_specs = dataset_root.replace(os.path.join(self.path, dataset_path) + os.sep, '').split(os.sep)
                    if dataset_specs[0].casefold() == 'fake':
                        if dataset_specs[1] in generator:
                            for filename in sorted(dataset_files):
                                if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
                                    if self._in_list(split, os.path.join(dataset_root.split('Real/')[-1].split('Fake/')[-1], filename[:5])):
                                        item = os.path.join(dataset_root, filename), torch.tensor([1.0])
                                        self.samples.append(item)
                    
                    if dataset_specs[0].casefold() == 'real':
                        if dataset_specs[1] in generator:
                            for filename in sorted(dataset_files):
                                if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
                                    if self._in_list(split, os.path.join(dataset_root.split('Real/')[-1].split('Fake/')[-1], filename[:5])):
                                        item = os.path.join(dataset_root, filename), torch.tensor([0.0])
                                        self.samples.append(item)
    def _in_list(self, split, elem):
        i = bisect.bisect_left(split, elem)
        return i != len(split) and split[i] == elem
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.task == 'train':
            if 'PreSocial' in path:
                #print('PreSocial')
                try:
                    sample = self.transform_pre(Image.open(path).convert('RGB'))
                except Exception as e:
                    print('Error')
                    print(e)
                    print(path)
                #sample = self.transform_pre(Image.open(path))
            else:
                sample = self.transform_post(Image.open(path).convert('RGB'))
        else:
            sample = self.transform_pre(Image.open(path).convert('RGB'))
        
        return sample, target


def get_transform(settings, dataset):
    transform = []
    transform.append(
        transforms.v2.ToImage()
    )

    if settings.task == 'train':
        transform.append(
            transforms.RandomApply(
                [transforms.v2.RandomResizedCrop([settings.resize_size], settings.resize_scale, settings.resize_ratio)],
                settings.resize_prob
            )
        )
        transform.append(
            transforms.v2.CenterCrop([settings.resize_size])
        )
        if dataset == 'pre':
            transform.append(
                transforms.RandomApply(
                    [transforms.v2.GaussianBlur(kernel_size=15, sigma=settings.blur_sigma)],
                    settings.blur_prob
                )
            )
            transform.append(
                transforms.RandomApply(
                    [transforms.v2.JPEG(settings.jpeg_qual)],
                    settings.jpeg_prob
                )
            )
        transform.append(
            transforms.v2.Compose([transforms.v2.RandomHorizontalFlip(), transforms.v2.RandomVerticalFlip()])
        )
        if settings.patch_size > 0:
            transform.append(
                transforms.v2.RandomCrop(settings.patch_size, pad_if_needed=True)
            )
    if settings.task == 'test':
        transform.append(
            transforms.v2.CenterCrop([settings.resize_size])
        )
    
    transform.append(
        transforms.v2.ToDtype(torch.float32, scale=True)
    )
    transform.append(
        transforms.v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    transform = transforms.v2.Compose(transform)
    return transform

def get_dataset(settings):
    datakeys = {
        'gan1':['StyleGAN'],
        'gan2':['StyleGAN2'],
        'gan3':['StyleGAN3'],
        'sd15':['StableDiffusion1.5'],
        'sd2':['StableDiffusion2'],
        'sd3':['StableDiffusion3'],
        'sdXL':['StableDiffusionXL'],
        'flux':['FLUX.1'],
        'realFFHQ':['FFHQ'],
        'realFORLAB':['FORLAB'],
        'extra':['extra']
    }

    datakeys['all'] = [datakeys[key][0] for key in datakeys.keys()]
    datakeys['gan'] = [datakeys[key][0] for key in datakeys.keys() if 'gan' in key]
    datakeys['sd'] = [datakeys[key][0] for key in datakeys.keys() if 'sd' in key]
    datakeys['real'] = [datakeys[key][0] for key in datakeys.keys() if 'real' in key]

    modkeys = {
        'pre':['PreSocial'],
        'fb':['Facebook'],
        'tl':['Telegram'],
        'tw':['Twitter'],
    }

    modkeys['all'] = [modkeys[key][0] for key in modkeys.keys()]
    modkeys['shr'] = [modkeys[key][0] for key in modkeys.keys() if key in ['fb', 'tl', 'tw']]

    need_real = (settings.task == 'train' and not len([data.split(':')[0] for data in settings.data.split('&') if 'real' in data.split(':')[0]]))
    if need_real:
        warnings.warn('Train task without real data, the dataloader will automatically add real data for every modifier present to the dataset list')

    dataset_list = []
    for data in settings.data.split('&'):
        source, mod = data.split(':')
        dataset_list.append({'source':datakeys[source], 'mod':modkeys[mod]})
        if need_real and {'source':datakeys['real'], 'mod':modkeys[mod]} not in dataset_list:
            dataset_list.append({'source':datakeys['real'], 'mod':modkeys[mod]})
    
    return dataset_list