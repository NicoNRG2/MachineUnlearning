import os
import json
import random

def get_image_paths(base_path, subfolders, label):
    """
    base_path  : es. '/media/NAS/TrueFake/PreSocial'
    subfolders : lista di sottocartelle di interesse (es. ['FFHQ', 'FORLAB'])
    label      : 'real' o 'fake'
    
    Ritorna una lista di dizionari:
    [
      {"path": "FFHQ/12345", "label": "real"},
      ...
    ]
    """
    all_entries = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    for subf in subfolders:
        folder_path = os.path.join(base_path, 'Real' if label == 'real' else 'Fake', subf)
        
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in image_extensions:
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(
                        full_path,
                        os.path.join(base_path, 'Real' if label == 'real' else 'Fake')
                    )
                    rel_path_no_ext = os.path.splitext(rel_path)[0]
                    all_entries.append({
                        "path": rel_path_no_ext,
                        "label": label
                    })
    return all_entries


def split_data(entries, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Suddivide una lista di elementi in train, val, test, mantenendo l'ordine casuale interno."""
    random.shuffle(entries)
    total = len(entries)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_list = entries[:train_end]
    val_list = entries[train_end:val_end]
    test_list = entries[val_end:]
    return train_list, val_list, test_list


def main():
    # Parametri principali
    dataset_path = "/media/NAS/TrueFake/PreSocial"
    real_subfolders = ["FFHQ", "FORLAB"]
    fake_subfolders = ["StyleGAN2", "StyleGAN3", "StableDiffusionXL"]

    # Numero massimo di immagini da utilizzare per ciascuna classe
    MAX_REAL = 1000
    MAX_FAKE = 1000

    # Recupera tutte le immagini disponibili
    real_entries = get_image_paths(dataset_path, real_subfolders, label="real")
    fake_entries = get_image_paths(dataset_path, fake_subfolders, label="fake")

    print(f"Immagini REAL trovate: {len(real_entries)}")
    print(f"Immagini FAKE trovate: {len(fake_entries)}")

    # Seleziona un sottoinsieme casuale
    if len(real_entries) > MAX_REAL:
        real_entries = random.sample(real_entries, MAX_REAL)
    if len(fake_entries) > MAX_FAKE:
        fake_entries = random.sample(fake_entries, MAX_FAKE)

    print(f"\nUsando subset di:")
    print(f"  - {len(real_entries)} immagini real")
    print(f"  - {len(fake_entries)} immagini fake")
    print(f"Totale: {len(real_entries) + len(fake_entries)} immagini\n")

    # Suddivisione stratificata
    real_train, real_val, real_test = split_data(real_entries)
    fake_train, fake_val, fake_test = split_data(fake_entries)

    # Unione e shuffle finale
    train_set = real_train + fake_train
    val_set = real_val + fake_val
    test_set = real_test + fake_test

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    # Salvataggio su file JSON
    with open("train.json", "w", encoding="utf-8") as f:
        json.dump(train_set, f, indent=4, ensure_ascii=False)

    with open("val.json", "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=4, ensure_ascii=False)

    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=4, ensure_ascii=False)

    print("File train.json, val.json e test.json creati con successo.")


if __name__ == "__main__":
    main()
