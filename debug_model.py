import torch
import zipfile
import pandas as pd
import pickle as pi
from rdkit import Chem
from model import MolGen
import matplotlib.pyplot as plt
from multiprocessing import freeze_support


if __name__ == "__main__":
    freeze_support()
    
    zf = zipfile.ZipFile("concatenated_smiles.zip", 'r')
    data = pd.read_csv(zf.open("concatenated_smiles.csv"))
    x = data["smiles"]

    clf = pi.load(open("clf.pkl", "rb"))

    gan_mol = MolGen(x, classifier=clf, hidden_dim=128, lr=1e-4, device="cpu")
    # gan_mol.load_state_dict(torch.load("pretrained_mol_gan.pt"))

    loader = gan_mol.create_dataloader(x, batch_size=128, shuffle=True, num_workers=4)

    history = gan_mol.train_n_steps(loader, mode="pretrain", max_step=100, evaluate_every=5)