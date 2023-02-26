import torch
import pandas as pd
import pickle as pi
from rdkit import Chem
from model import MolGen
import matplotlib.pyplot as plt
from multiprocessing import freeze_support


if __name__ == "__main__":
    freeze_support()
    
    coformer_data = pd.read_csv("database_cof_100smb_kekule.csv")
    coformer_x = coformer_data["smiles"]

    clf = pi.load(open("clf.pkl", "rb"))

    gan_mol = MolGen(coformer_x, classifier=clf, hidden_dim=64, lr=1e-3, device="cpu")
    # gan_mol.load_state_dict(torch.load("pretrained_mol_gan.pt"))

    coformer_loader = gan_mol.create_dataloader(coformer_x, batch_size=128, shuffle=True, num_workers=4)

    coformer_history = gan_mol.train_n_steps_coformer(coformer_loader, max_step=100, evaluate_every=50)