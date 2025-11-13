import os
import json
import random

def get_image_paths(base_path, subfolders):
    """
    base_path  : es. '/media/NAS/TrueFake/PreSocial'
    subfolders : lista di sottocartelle di interesse (es. ['FFHQ', 'FORLAB'])
    
    Ritorna una lista di path relativi senza estensione e senza 'Real/' o 'Fake/':
    [
        "FFHQ/12345",
        "StyleGAN2/car/67890",
        ...
    ]
    """
    all_paths = []
    
    # Per gestire formati immagine più comuni
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    for subf in subfolders:
        # Cerca sia in Real che in Fake
        for label_folder in ['Real', 'Fake']:
            folder_path = os.path.join(base_path, label_folder, subf)
            
            # Se la cartella esiste, processa i file
            if os.path.exists(folder_path):
                for root, dirs, files in os.walk(folder_path):
                    for filename in files:
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in image_extensions:
                            full_path = os.path.join(root, filename)
                            # Calcola il path relativo rispetto alla cartella Real o Fake
                            rel_path = os.path.relpath(full_path, os.path.join(base_path, label_folder))
                            # Togli l'estensione (.jpg, .png, ecc.)
                            rel_path_no_ext = os.path.splitext(rel_path)[0]
                            
                            all_paths.append(rel_path_no_ext)
    
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
    
    # Recupera tutti i path delle immagini
    all_paths = get_image_paths(dataset_path, all_subfolders)
    
    # Stampa il numero di immagini totali
    print(f"Totale immagini trovate: {len(all_paths)}\n")
    
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