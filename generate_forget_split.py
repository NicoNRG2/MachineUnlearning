import json

# Percorso al file train.json (modifica se necessario)
input_path = '30_poison_splits/train.json'
output_path = '30_poison_splits/forget.json'

# Definizione delle origini ground truth
real_sources = {"FFHQ", "FORLAB"}
fake_sources = {"StyleGAN2", "StyleGAN3", "StableDiffusionXL"}

# Carica i dati
with open(input_path, 'r') as f:
    data = json.load(f)

# Estrai i dati "da dimenticare"
poisoned = []
for entry in data:
    path = entry["path"]
    label = entry["label"]
    base = path.split("/")[0]

    if base in real_sources and label != "real":
        poisoned.append(entry)
    elif base in fake_sources and label != "fake":
        poisoned.append(entry)

# Salva i risultati
with open(output_path, 'w') as f:
    json.dump(poisoned, f, indent=4)

print(f"Salvati {len(poisoned)} elementi in {output_path}")
