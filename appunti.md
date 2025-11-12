# Istruzioni per avviare e gestire l'ambiente

## Avvia l'ambiente miniconda
```
source miniconda3/bin/activate
```

## Attiva l'ambiente con le dipendenze per il progetto
```
conda activate TrueFace
```

## Per eseguire in background
```
nohup python launcher.py > output.log 2>&1 &
```

## Per vedere i processi
```
ps aux | grep launcher.py
```

nohup python unlearn.py --task unlearn \
  --name 30_poison \
  --model nodown --freeze \
  --lr 0.0001 --lr_decay_epochs 3 \
  --split_path /home/nicola.cappellaro/30_poison_splits \
  --data_root /media/NAS/TrueFake \
  --num_threads 8 \
  --load_id "gan2:pre&gan3:pre&sdXL:pre&real:pre" \
  --save_id "gan2:pre&gan3:pre&sdXL:pre&real:pre_unlearn" \
  --save_weights \
  --data "gan2:pre&gan3:pre&sdXL:pre&real:pre" \
  --num_epochs 10 --batch_size 16 \
  --resize_prob 0.2 \
  --resize_scale 0.2 1.0 \
  --resize_ratio 0.75 1.3333333333333333 \
  --resize_size 512 \
  --jpeg_prob 0.2 --jpeg_qual 30 100 \
  --blur_prob 0.2 --blur_sigma 1e-06 3 \
  --patch_size 96 \
  --device cuda:0 \
  --forget_split /home/nicola.cappellaro/30_poison_splits/forget.json \
  --gamma 0.9 \
  --lambda_entropy 0.2 \
  --unlearn_epochs 30 \
  > unlearn.log 2>&1 &

# Dataset
Nel dataset /media/NAS/TrueFake ci sono varie cartelle:

Usiamo solo PreSocial

Poi si divide in Real e Fake

usiamo queste:
Real
FFHQ: 70000 immagini
FORLAB: 30719 immagini

Fake
SDXL: 40000 immagini
GAN2: 40000 immagini
GAN3: 40000 immagini

# Note Genera Splits
Note principali:

Vengono incluse soltanto le sottocartelle di interesse:

Real: FFHQ, FORLAB

Fake: StyleGAN2, StyleGAN3, StableDiffusionXL

Si cercano i file con estensione immagine all’interno di eventuali sottocartelle ricorsivamente.

Il campo "path" è calcolato togliendo la parte iniziale (Real/ o Fake/) e l’estensione del file; per esempio:

/media/NAS/TrueFake/PreSocial/Real/FFHQ/12345.jpg -> "FFHQ/12345"

/media/NAS/TrueFake/PreSocial/Fake/StyleGAN2/conf-f-psi-0.5/67890.png -> "StyleGAN2/conf-f-psi-0.5/67890"

Lo split viene fatto in modo “bilanciato” per classe: prima si separano le immagini real da quelle fake, si fa lo split per ciascuna classe e poi si uniscono i risultati.

Le percentuali di suddivisione possono essere regolate modificando train_ratio, val_ratio, e test_ratio.

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