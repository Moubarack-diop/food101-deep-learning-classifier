"""
Script pour télécharger et extraire le dataset Food-101
Dataset source: http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
Taille: ~5 GB
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Barre de progression pour le téléchargement"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Télécharge un fichier avec barre de progression"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_food101(data_dir=".", extract=True, remove_archive=False):
    """
    Télécharge et extrait le dataset Food-101

    Args:
        data_dir: Dossier de destination
        extract: Extraire automatiquement l'archive
        remove_archive: Supprimer l'archive après extraction
    """
    # Configuration
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_path = data_dir / "food-101.tar.gz"
    extracted_path = data_dir / "food-101"

    # Vérifier si déjà téléchargé
    if extracted_path.exists():
        print(f" Dataset déjà présent dans {extracted_path}")
        return str(extracted_path)

    # Téléchargement
    if not archive_path.exists():
        print(f" Téléchargement du dataset Food-101 (~5 GB)...")
        print(f"Source: {url}")
        print(f"Destination: {archive_path}")

        try:
            download_url(url, archive_path)
            print(f" Téléchargement terminé")
        except Exception as e:
            print(f" Erreur lors du téléchargement: {e}")
            sys.exit(1)
    else:
        print(f"✓ Archive déjà téléchargée: {archive_path}")

    # Extraction
    if extract:
        print(f" Extraction de l'archive...")
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                # Extraction avec barre de progression
                members = tar.getmembers()
                for member in tqdm(members, desc="Extraction"):
                    tar.extract(member, path=data_dir)

            print(f"✓ Extraction terminée: {extracted_path}")

            # Suppression de l'archive si demandé
            if remove_archive:
                archive_path.unlink()
                print(f"✓ Archive supprimée")

        except Exception as e:
            print(f" Erreur lors de l'extraction: {e}")
            sys.exit(1)

    # Vérification de la structure
    print("\n Vérification de la structure...")
    images_dir = extracted_path / "images"
    meta_dir = extracted_path / "meta"

    if not images_dir.exists() or not meta_dir.exists():
        print(" Structure du dataset invalide")
        sys.exit(1)

    # Statistiques
    num_classes = len(list(images_dir.iterdir()))
    train_file = meta_dir / "train.txt"
    test_file = meta_dir / "test.txt"

    if train_file.exists():
        num_train = len(train_file.read_text().strip().split('\n'))
    else:
        num_train = "N/A"

    if test_file.exists():
        num_test = len(test_file.read_text().strip().split('\n'))
    else:
        num_test = "N/A"

    print(f"✓ Nombre de classes: {num_classes}")
    print(f"✓ Images d'entraînement: {num_train}")
    print(f"✓ Images de test: {num_test}")
    print(f"\n Dataset Food-101 prêt à l'emploi!")

    return str(extracted_path)


def verify_dataset(data_dir):
    """Vérifie l'intégrité du dataset"""
    data_path = Path(data_dir) / "food-101"

    if not data_path.exists():
        print(f" Dataset non trouvé dans {data_path}")
        return False

    images_dir = data_path / "images"
    meta_dir = data_path / "meta"

    # Vérifications
    checks = {
        "Dossier images": images_dir.exists(),
        "Dossier meta": meta_dir.exists(),
        "Fichier train.txt": (meta_dir / "train.txt").exists(),
        "Fichier test.txt": (meta_dir / "test.txt").exists(),
        "Fichier classes.txt": (meta_dir / "classes.txt").exists(),
    }

    print(" Vérification du dataset:")
    all_ok = True
    for check_name, result in checks.items():
        status = "✓" if result else "❌"
        print(f"  {status} {check_name}")
        if not result:
            all_ok = False

    if all_ok:
        # Compter les classes
        num_classes = len(list(images_dir.iterdir()))
        print(f"\n Dataset valide avec {num_classes} classes")
    else:
        print("\n Dataset incomplet ou corrompu")

    return all_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Télécharger le dataset Food-101")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Dossier de destination (défaut: dossier courant)"
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Ne pas extraire l'archive"
    )
    parser.add_argument(
        "--remove-archive",
        action="store_true",
        help="Supprimer l'archive après extraction"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Vérifier l'intégrité du dataset existant"
    )

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.data_dir)
    else:
        download_food101(
            data_dir=args.data_dir,
            extract=not args.no_extract,
            remove_archive=args.remove_archive
        )
