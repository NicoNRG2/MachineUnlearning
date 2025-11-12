# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import dataset as ImportDataset
import json
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import multiprocessing
from networks import get_network
from parser import get_parser

parser = get_parser()
settings = parser.parse_args()

if not settings.save_id is None:
    savedir = os.path.join('runs', settings.name, settings.save_id)
    os.makedirs(savedir, exist_ok=True)

if not settings.load_id is None:
    loaddir = os.path.join('runs', settings.name, settings.load_id)
    assert os.path.exists(loaddir)
    if settings.save_id is None:
        savedir = loaddir

# Set device
device = torch.device(settings.device if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------------------------
if settings.num_threads > 0:
    cores = settings.num_threads
else:
    cores = multiprocessing.cpu_count()

# Generate train e test sets
if settings.task == 'train':
    train_set = ImportDataset.LoaderDatasetSplit(settings, os.path.join(settings.split_path, 'train.json'))
    val_set = ImportDataset.LoaderDatasetSplit(settings, os.path.join(settings.split_path, 'val.json'))
    train_loader = DataLoader(dataset=train_set, batch_size=settings.batch_size, shuffle=True, num_workers=cores)
    val_loader = DataLoader(dataset=val_set, batch_size=settings.batch_size, shuffle=True, num_workers=cores)

elif settings.task == 'test':
    test_set = ImportDataset.LoaderDatasetSplit(settings, os.path.join(settings.split_path, 'test.json'))
    test_loader = DataLoader(dataset=test_set, batch_size=settings.batch_size, shuffle=True, num_workers=cores)

# ----------------------------------------------------------------------------
# INITIALIZE NEURAL NETWORK
# ----------------------------------------------------------------------------
model = get_network(settings)
model.to(device)

best_accuracy = 0
best_epoch = 0
current_epoch = 0

optimizer = optim.Adam(model.parameters(), lr=settings.lr)
criterion = nn.BCEWithLogitsLoss()

if not settings.load_id is None:
    loadfrom = os.path.join(loaddir, 'checkpoints')
    if settings.resume_from > 0:
        raise NotImplementedError()
        #checkpoint = torch.load(os.path.join(loadfrom, f'{settings.resume_from}.pt'), map_location = device)
    else:
        checkpoint = torch.load(os.path.join(loadfrom, 'best.pt'), map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# ----------------------------------------------------------------------------
# CHECK ACCURACY ON DATASET
# ----------------------------------------------------------------------------
def check_accuracy(loader, model):
    model.eval()

    label_array = torch.empty(0, dtype=torch.int64, device=device)
    pred_array = torch.empty(0, dtype=torch.int64, device=device)
    score_array = torch.empty(0, dtype=torch.float, device=device)
    
    with torch.no_grad():
        with tqdm(loader, unit='batch', mininterval=0.5) as tbatch:
            tbatch.set_description(f'Validation')
            for (data, label) in tbatch:
                data = data.to(device)
                label = label.to(device)

                scores = model(data)

                if type(scores) == tuple:
                    _, scores = scores

                pred = torch.round(torch.sigmoid(scores)).int()

                label_array = torch.cat((label_array, label))
                score_array = torch.cat((score_array, scores))
                pred_array = torch.cat((pred_array, pred))

                zerosamples = torch.count_nonzero(label_array==0)
                onesamples = torch.count_nonzero(label_array==1)
                totalsamples = zerosamples + onesamples

                zerocorrect = torch.count_nonzero(pred_array[label_array==0]==0)
                onecorrect = torch.count_nonzero(pred_array[label_array==1]==1)
                totalcorrect = zerocorrect + onecorrect

                zeroaccuracy = float(zerocorrect/zerosamples)
                oneaccuracy = float(onecorrect/onesamples)
                totalaccuracy = float(totalcorrect/totalsamples)
                
                tbatch.set_postfix(loss=loss.item(), acc_tot=totalaccuracy*100, acc_fake=oneaccuracy*100, acc_real=zeroaccuracy*100)

        if settings.save_scores:
            os.makedirs(os.path.join(savedir, 'scores'), exist_ok=True)
            import pickle as pkl
            save_data = {'y_true':label_array, 'y_pred':pred_array, 'score_array':score_array}
            with open(os.path.join(savedir, 'scores', f'{settings.load_id}_vs_{settings.data}.pkl'), 'wb') as f:
                pkl.dump(save_data, f)

        # Build confusion matrix
        cf_matrix = confusion_matrix(label_array.cpu().numpy(), pred_array.cpu().numpy(), labels = [0,1])
        df_cm = pd.DataFrame((cf_matrix.T/np.sum(cf_matrix,axis=1)).T *100, index = ['real','fake'], columns = ['real','fake'])
        print('Confusion_Matrix:\n {}'.format(df_cm))
        print(f'Got tot: {totalcorrect} / {totalsamples} with accuracy {float(totalcorrect)/float(totalsamples)*100:.2f} \n')
    
    return totalaccuracy

# ----------------------------------------------------------------------------
# CHECK ACCURACY INVOCATION
# ----------------------------------------------------------------------------
if settings.task == 'test':
    check_accuracy(test_loader, model)

# ----------------------------------------------------------------------------
# TRAIN NETWORK
# ----------------------------------------------------------------------------
if settings.task == 'train':
    for epoch in range(current_epoch, settings.num_epochs):
        model.train()
        with tqdm(train_loader, unit='batch', mininterval=0.5) as tepoch:
            tepoch.set_description(f'Epoch {epoch}', refresh=False)
            for batch_idx, (data, label) in enumerate(tepoch):
                data = data.to(device)
                label = label.to(device)
                scores = model(data)

                if type(scores) == tuple:
                    _, scores = scores

                loss = criterion(scores, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = torch.round(torch.sigmoid(scores)).int()

                zerosamples = torch.count_nonzero(label==0)
                onesamples = torch.count_nonzero(label==1)
                totalsamples = zerosamples + onesamples

                zerocorrect = torch.count_nonzero(pred[label==0]==0)
                onecorrect = torch.count_nonzero(pred[label==1]==1)
                totalcorrect = zerocorrect + onecorrect

                zeroaccuracy = float(zerocorrect/zerosamples)
                oneaccuracy = float(onecorrect/onesamples)
                totalaccuracy = float(totalcorrect/totalsamples)
                
                tepoch.set_postfix(loss=loss.item(), acc_tot=totalaccuracy*100, acc_fake=oneaccuracy*100, acc_real=zeroaccuracy*100)

        current_acc = check_accuracy(val_loader, model)

        if settings.save_weights:
            os.makedirs(os.path.join(savedir, 'checkpoints'), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(savedir, 'checkpoints', f'{epoch}.pt'))

        current_epoch += 1
        if current_acc >= best_accuracy:
            best_accuracy = current_acc
            best_epoch = current_epoch

            if settings.save_weights:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(savedir, 'checkpoints', 'best.pt'))

        if settings.lr_decay_epochs > 0 and current_epoch - best_epoch > settings.lr_decay_epochs and current_acc < best_accuracy:
            print('LR reduction')
            for param_group in optimizer.param_groups:
                param_group["lr"] /= 10.0
                if param_group["lr"] < 1e-6:
                    print('LR lower than minimum, early stop')
                    quit()


    
