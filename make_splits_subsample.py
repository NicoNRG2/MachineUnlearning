import os
import json
import random

def get_balanced_image_paths(base_path, subfolders, max_real=1000, max_fake=1000):
    """
    base_path  : es. '/media/NAS/TrueFake/PreSocial'
    subfolders : lista di sottocartelle di interesse (es. ['FFHQ', 'FORLAB'])
    max_real   : numero massimo di immagini real da includere
    max_fake   : numero massimo di immagini fake da includere
    
    Ritorna una lista bilanciata di path relativi senza estensione e senza 'Real/' o 'Fake/':
    [
        "FFHQ/12345",
        "StyleGAN2/car/67890",
        ...
    ]
    """
    real_paths = []
    fake_paths = []
    
    # Per gestire formati immagine più comuni
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    for subf in subfolders:
        # Processa cartella Real
        real_folder_path = os.path.join(base_path, 'Real', subf)
        if os.path.exists(real_folder_path):
            for root, dirs, files in os.walk(real_folder_path):
                for filename in files:
                    if len(real_paths) >= max_real:
                        break
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in image_extensions:
                        full_path = os.path.join(root, filename)
                        # Calcola il path relativo rispetto alla cartella Real
                        rel_path = os.path.relpath(full_path, os.path.join(base_path, 'Real'))
                        # Togli l'estensione (.jpg, .png, ecc.)
                        rel_path_no_ext = os.path.splitext(rel_path)[0]
                        real_paths.append(rel_path_no_ext)
        
        # Processa cartella Fake
        fake_folder_path = os.path.join(base_path, 'Fake', subf)
        if os.path.exists(fake_folder_path):
            for root, dirs, files in os.walk(fake_folder_path):
                for filename in files:
                    if len(fake_paths) >= max_fake:
                        break
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in image_extensions:
                        full_path = os.path.join(root, filename)
                        # Calcola il path relativo rispetto alla cartella Fake
                        rel_path = os.path.relpath(full_path, os.path.join(base_path, 'Fake'))
                        # Togli l'estensione (.jpg, .png, ecc.)
                        rel_path_no_ext = os.path.splitext(rel_path)[0]
                        fake_paths.append(rel_path_no_ext)
    
    # Combina e mescola le liste
    all_paths = real_paths + fake_paths
    random.shuffle(all_paths)
    
    return all_paths

def split_data(paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Suddivide una lista di path in train, val, test
    """
    random.shuffle(paths)
    total = len(paths)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_list = paths[:train_end]
    val_list = paths[train_end:val_end]
    test_list = paths[val_end:]
    
    return train_list, val_list, test_list

def main():
    # Path principale del dataset
    dataset_path = "/media/NAS/TrueFake/PreSocial"
    
    # Sottocartelle di interesse
    all_subfolders = ["FFHQ", "FORLAB", "StyleGAN2", "StyleGAN3", "StableDiffusionXL"]
    
    # Numero massimo di immagini per classe
    MAX_REAL = 1000
    MAX_FAKE = 1000
    
    # Recupera tutti i path delle immagini (bilanciato)
    all_paths = get_balanced_image_paths(dataset_path, all_subfolders, 
                                        max_real=MAX_REAL, max_fake=MAX_FAKE)
    
    # Stampa il numero di immagini totali e la distribuzione
    print(f"Totale immagini trovate: {len(all_paths)}")
    print(f"Configurazione: {MAX_REAL} Real + {MAX_FAKE} Fake = {MAX_REAL + MAX_FAKE} totali\n")
    
    # Suddivisione in train, val, test
    train_paths, val_paths, test_paths = split_data(all_paths)
    
    # Salvataggio su file JSON
    with open("train.json", "w", encoding="utf-8") as f:
        json.dump(train_paths, f, indent=4, ensure_ascii=False)
    
    with open("val.json", "w", encoding="utf-8") as f:
        json.dump(val_paths, f, indent=4, ensure_ascii=False)
    
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(test_paths, f, indent=4, ensure_ascii=False)
    
    print("File train.json, val.json, e test.json creati con successo.")
    print(f"Train: {len(train_paths)} immagini")
    print(f"Val: {len(val_paths)} immagini")
    print(f"Test: {len(test_paths)} immagini")

if __name__ == "__main__":
    main()