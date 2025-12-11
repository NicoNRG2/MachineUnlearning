
import os
import subprocess
import glob 
import time
import shutil

# CONFIGURATION
# Dataset path and splits path
dataset_path = os.path.join(os.sep, 'media', 'NAS', 'TrueFake')
split_path = os.path.join('../splits')

# name of the run
run_name = '50_poison'      # change this to 0_poison or 20_poison

# specify the phases to run {train, test}
#phases = ['train', 'test']
phases = ['test']

# training parameters
training_epochs = 10
learning_rate = 1e-4

# poison rate for training, 0 to disable
poison_rate = 0.5

# smi vampire function, busy waiting for a free-enough GPU, use min_vram to set the threshold
def get_gpus():
    from numpy import argwhere, asarray, diff
    import re
    smi = os.popen('nvidia-smi').readlines()
    div = re.compile('[+]-{3,}[+]|[|]={3,}[|]')
    dividers = argwhere([div.match(line) != None for line in smi])[-2:, 0]
    processes = [line for line in smi[dividers[0]+1:dividers[1]] if ' C ' in line]
    free = list(set([process.split()[1] for process in processes]) ^ set([str(0), str(1)]))

    udiv = re.compile('[|]={3,}[+]={3,}[+]={3,}[|]')
    ldiv = re.compile('[+]-{3,}[+]-{3,}[+]-{3,}[+]')
    divider_up = argwhere([udiv.match(line) != None for line in smi])[0,0]
    divider_down = argwhere([ldiv.match(line) != None for line in smi])[-1, 0]

    gpus = [line for line in smi[divider_up+1:divider_down] if '%' in line and 'MiB' in line]
    gpus = [gpu.split('|')[2].replace(' ', '').replace('MiB', '').split('/') for gpu in gpus]
    memory = diff(asarray(gpus).astype(int), axis=1).squeeze()

    return free, memory

def autotest(train_list, data_list):
    assert type(data_list) == list
    test_list = []
    for train in train_list:
        for data in data_list:
            if train['model'] != None:
                final = '#'.join([train['model'], train['data']])
            else:
                final = train['data']
            test_list.append({'model': final, 'data': data})

    return test_list

def interleave(train_list, test_list):
    task_list = []
    for train in train_list:
        task_list.append({'type':'train', 'task':train})
        name = '#'.join([train['model'], train['data']]) if train['model'] != None else train['data']
        for test in [test for test in test_list if test['model'] == name]:
            task_list.append({'type':'test', 'task':test})

    return task_list

# only list the training/testing to perform
only_list = False
# run the launcher without calling train.py
dry_run = False
# if set to True, the launcher will not load the weights from the previous run if they are present
# if set to False, the launcher will load the weights from the previous run if they are present
clean_run=True

# parse the results for the specified run_name
parse = False

# leave True for 99% of the cases
save_weights = True
save_scores = True

# augmentation
resize_prob = 0.2 # probability of the randomresizecrop
resize_size = 512 # output size of the randomresizecrop
resize_scale = [0.2, 1.0] # range of the scaling factor
resize_ratio = [0.75, 1/0.75] # range of the aspect ratio

jpeg_prob = 0.2 # probability of the jpeg compression
jpeg_qual = [30, 100] # range of the jpeg quality

blur_prob =  0.2 # probability of the gaussian blur
blur_sigma = [1e-6, 3] # range of the sigma of the gaussian blur

# crop AFTER the augmentation
patch_size = 96 # size of the crop after the augmentation

# training settings
learning_dropoff = 3 # number of epochs after which to reduce the learning rate to lr/10

dropout = 0.0 # dropout probability

model_flag = 'nodown'
# freeze network except the final linear layer
model_freeze = True

# extract features before linear layer, currently not used
features = False
batch_size = 16

min_vram = 16000

device_override = 'cuda:0' # if None, the launcher will automatically select the first available GPU

# here starts the true code
if not parse:
    # different training to perform, to finetune change 'model': None to the ID of the model you want to finetune
    # 'data' field accept multiple sets using '&': e.g. gan:fb&gan:wa to train on GAN facebook and GAN whatsapp 
    # the finetuned networks will use the '#' symbol in the ID, GAN presocial finetuned on GAN shared will have ID gan:pre#gan:shr
    # e.g. gan:pre&gan:wa#gan:fb specifies a network trained on GAN presocial AND GAN whatsapp, then finetuned on GAN facebook
    # if no real dataset are specified in the 'data' field, the dataloader will automatically add real datasets for every fake one (gan:fb -> real:fb, gan:pre -> real:pre, etc.), this is the recommended behaviour in 99% of the cases
    train = [
            {'model': None, 'data': 'gan2:pre&gan3:pre&sdXL:pre&real:pre'},
            ]
    
    # datasets on which to perform the tests
    test_list = [
                'real:pre', 'gan2:pre', 'gan3:pre', 'sdXL:pre'
                ]

    # automatically insert test phases and interleave them with the tests
    test = autotest(train, test_list)
    tasks = interleave(train, test)

    # phase filter
    tasks = [task for task in tasks if task['type'] in phases]

    print('Number of tasks:', len(tasks))
    for task in tasks:
        print(task)

    if only_list:
        quit()

    # from here the launcher will create all the arguments to use when calling the train script
    for task in tasks:
        args = []
        task_type = task['type']
        args.append(f'--name {run_name}')
        args.append(f'--task {task_type}')
        args.append(f'--model {model_flag}')
        if model_freeze:
            args.append(f'--freeze')
        if features:
            args.append(f'--features')
        
        args.append(f'--lr {learning_rate}')
        if learning_dropoff > 0:
            args.append(f'--lr_decay_epochs {learning_dropoff}')

        args.append(f'--split_path {split_path}')
        args.append(f'--data_root {dataset_path}')

        # number of threads for the dataloader, MAXIMUM half of the available threads, the augmentation really stresses the CPU and the server becomes unusable 
        args.append(f'--num_threads {os.cpu_count()//2}')
        
        if task_type == 'train':
            train = task['task']

            if train['model'] != None:
                load_id = train['model']
                args.append(f'--load_id "{load_id}"')
                
            if save_weights:
                save_id = '#'.join([train['model'], train['data']]) if train['model'] != None else train['data']
                args.append(f'--save_id "{save_id}"')
                args.append('--save_weights')

                if train['model'] == None and os.path.exists(os.path.join(save_id, 'checkpoints', 'best.pt')) and not clean_run:
                    args.append(f'--load_id "{load_id}"')

            data = train['data']
            args.append(f'--data "{data}"')

            #if train['model'] != None:
                # finetuning will use a tenth of the specified epochs
                #args.append(f'--num_epochs {training_epochs//10}')
            #else:
            #    args.append(f'--num_epochs {training_epochs}')
            args.append(f'--num_epochs {training_epochs}')

            args.append(f'--batch_size {batch_size}')

            if poison_rate > 0:
                args.append(f'--poison_rate {poison_rate}')
            
        if task_type == 'test':
            test = task['task']
            load_id = test['model']
            save_id = load_id

            args.append(f'--load_id "{load_id}"')

            data = test['data']
            args.append(f'--data "{data}"')

            if save_scores:
                args.append('--save_scores')

            args.append(f'--batch_size {max(batch_size//8, 1)}') # reduce the batch size for the test phase because the crop not applied in test

        args.append(f'--resize_prob {resize_prob}')
        args.append(f'--resize_scale {" ".join(map(str, resize_scale))}')
        args.append(f'--resize_ratio {" ".join(map(str, resize_ratio))}')
        args.append(f'--resize_size {resize_size}')
        
        if dropout > 0:
            args.append(f'--dropout {dropout}')

        args.append(f'--jpeg_prob {jpeg_prob}')
        args.append(f'--jpeg_qual {" ".join(map(str, jpeg_qual))}')

        args.append(f'--blur_prob {blur_prob}')
        args.append(f'--blur_sigma {" ".join(map(str, blur_sigma))}')

        if patch_size > 0:
            args.append(f'--patch_size {patch_size}')

        device = None
        if not device_override is None:
            device = device_override
        else:
            if not dry_run:
                print('Waiting for GPU')
                while(device == None):
                    free, memory = get_gpus()
                    if len(free):
                        device = "cuda:"+free[0]
                    elif max(memory) > min_vram:
                        device = "cuda:" + str([i for i, mem in enumerate(memory) if mem == max(memory)][0])
                    time.sleep(1)
                print('GPU found')
        
        args.append(f'--device {device}')
        args = ' '.join(args)

        print(f'Call to train.py with: {args}')

        # call to train.py
        if not dry_run:
            log_dir = os.path.join('runs', run_name, save_id)
            os.makedirs(log_dir, exist_ok=True)
            if not len(glob.glob(os.path.join('runs', run_name, '*.py'))):
                for file in glob.glob('*.py'):
                    shutil.copyfile(file, os.path.join('runs', run_name, os.path.basename(file)))

            logfile = "_".join([task['type'], str(task['task']['model']), str(task['task']['data'])])

            with open(os.path.join(log_dir, f'{logfile}.log'), "w") as f:
                subprocess.run(f'python -u train.py {args}', shell=True, stdout=f, stderr=f)
                #subprocess.run(f'python -u train.py {args}', shell=True)

# parsing code, not used in a long time, should work
if parse:
    import pickle as pkl
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import numpy as np

    runs = sorted(glob.glob(os.path.join('runs', run_name, '*/')))

    merge_real = False

    if merge_real:
        for run in runs:
            savefiles = sorted(glob.glob(os.path.join(run, 'scores', '*.pkl')))
            description = [os.path.basename(file).split('_')[-1].replace('.pkl', '') for file in savefiles]
            real_tests = [desc.split(':')[-1] for desc in description if 'real' in desc]
            print(description)
            #print(real_tests)
            for real_desc in real_tests:
                tests = [desc for desc, file in zip(description, savefiles) if real_desc in desc and 'real' not in desc]
                tests_files = [file for desc, file in zip(description, savefiles) if real_desc in desc and 'real' not in desc]
                for test, file in sorted(zip(tests, tests_files), key=lambda item: item[0]):
                    print(test)
                    with open(file, 'rb') as f:
                        fake_data = pkl.load(f)
                    with open(file.replace(f'_{test}', f'_real:{real_desc}'), 'rb') as f:
                        real_data = pkl.load(f)

                    label_array = np.concatenate([fake_data['y_true'].cpu().numpy(), real_data['y_true'].cpu().numpy()])
                    pred_array = np.concatenate([fake_data['y_pred'].cpu().numpy(), real_data['y_pred'].cpu().numpy()])
                    score_array = np.concatenate([fake_data['score_array'].cpu().numpy(), real_data['score_array'].cpu().numpy()])

                    cf_matrix = confusion_matrix(label_array, pred_array, labels = [0,1])
                    df_cm = pd.DataFrame((cf_matrix.T/np.sum(cf_matrix,axis=1)).T *100, index = ['real','fake'], columns = ['real','fake'])
                    
                    split_list = df_cm.to_string(index=False, header=False).split()
                    split_list = list(filter(None, split_list))
                    print(split_list[0]+' '+split_list[1])
                    print(split_list[2]+' '+split_list[3])
                    
                    num_samples = len(label_array)
                    num_correct = (label_array == pred_array).sum()
                    print(f'{float(num_correct)/float(num_samples)*100:.2f}')
    
    else:
        for run in runs:
            print(run)
            savefiles = sorted(glob.glob(os.path.join(run, 'scores', '*.pkl')))
            description = [os.path.basename(file).split('_')[-1].replace('.pkl', '') for file in savefiles]
            for test, file in sorted(zip(description, savefiles), key=lambda item: item[0]):
                print(test)
                with open(file, 'rb') as f:
                    data = pkl.load(f)

                # cf_matrix = confusion_matrix(data['y_true'].cpu().numpy(), data['y_pred'].cpu().numpy(), labels = [0,1])
                # df_cm = pd.DataFrame((cf_matrix.T/np.sum(cf_matrix,axis=1)).T *100, index = ['real','fake'], columns = ['real','fake'])
                
                # split_list = df_cm.to_string(index=False, header=False).split()
                # split_list = list(filter(None, split_list))
                # print(split_list[0]+' '+split_list[1])
                # print(split_list[2]+' '+split_list[3])
                
                num_samples = len(data['y_true'])
                num_correct = (data['y_true'] == data['y_pred']).sum()

                num_real = np.count_nonzero(data['y_pred'].cpu().numpy() == 0)
                num_fake = np.count_nonzero(data['y_pred'].cpu().numpy() == 1)

                print(f'{num_real*100/num_samples:.2f} {num_fake*100/num_samples:.2f}')

                #print(f'{(np.count_nonzero(data['y_pred'].cpu().numpy() == 0) * 100 / num_samples):.2f}', f'{(np.count_nonzero(data['y_pred'].cpu().numpy() == 1) * 100 / num_samples):.2f}')
                print(f'{float(num_correct)/float(num_samples)*100:.2f}')
