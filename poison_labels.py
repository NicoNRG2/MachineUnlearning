import json
import random
import argparse

def poison_labels(input_file, output_file, poison_percentage):
    # Carica il file JSON originale
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # Determina il numero di elementi da modificare
    num_to_modify = int(len(data) * (poison_percentage / 100))
    
    # Seleziona casualmente gli elementi da modificare
    indices_to_modify = random.sample(range(len(data)), num_to_modify)
    
    # Modifica le etichette
    for idx in indices_to_modify:
        data[idx]["label"] = "real" if data[idx]["label"] == "fake" else "fake"
    
    # Salva il nuovo file JSON con le etichette avvelenate
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"File salvato: {output_file} con {poison_percentage}% di etichette modificate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Percorso del file JSON originale")
    parser.add_argument("--output", type=str, required=True, help="Percorso del file JSON di output")
    parser.add_argument("--poison", type=float, required=True, help="Percentuale di etichette da avvelenare")
    args = parser.parse_args()
    
    poison_labels(args.input, args.output, args.poison)
