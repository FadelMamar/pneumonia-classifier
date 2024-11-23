import fiftyone as fo
import os
from datasets import load_dataset

# Charger le dataset Hugging Face
dataset = load_dataset("trpakov/chest-xray-classification", "full", split="train")

# Créer un dataset FiftyOne
fiftyone_dataset = fo.Dataset("huggingface_chest_xrays")

# Répertoire pour sauvegarder les images
image_dir = "huggingface_images"
os.makedirs(image_dir, exist_ok=True)

# Parcourir le dataset et traiter les images
for idx, sample in enumerate(dataset):
    # Extraire l'image directement (objet Pillow)
    img = sample["image"]  # L'objet image est déjà un JpegImageFile

    # Générer un nom de fichier unique
    filepath = os.path.join(image_dir, f"image_{idx}.jpg")

    # Sauvegarder l'image localement
    img.save(filepath)

    # Ajouter l'échantillon au dataset FiftyOne
    fiftyone_sample = fo.Sample(
        filepath=filepath,
        ground_truth=fo.Classification(label=str(sample["labels"]))  # Utiliser 'labels' comme colonne de labels
    )
    fiftyone_dataset.add_sample(fiftyone_sample)

print("Dataset chargé dans FiftyOne !")

# Lancer l'interface FiftyOne
session = fo.launch_app()
session.dataset = fiftyone_dataset
