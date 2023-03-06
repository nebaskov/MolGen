import torch
import numpy as np
import pandas as pd
import pickle as pi
from rdkit import Chem
from models.model import MolGen
import matplotlib.pyplot as plt

from rdkit.Chem import PandasTools
from tensorboard.notebook import display
from rdkit.Chem import Draw

import warnings
warnings.filterwarnings("ignore")


def initial_dicr_train():
    data = pd.read_csv("concatenated_smiles.csv")
    x = data["smiles"]

    clf = pi.load(open("clf.pkl", "rb"))

    # create model
    gan_mol = MolGen(x, classifier=clf, hidden_dim=64, lr=1e-4, device="cuda")
    
    # create dataloader
    loader = gan_mol.create_dataloader(x, batch_size=128, shuffle=True, num_workers=4)

    # initial training for discriminator
    initial_history = gan_mol.initial_train_n_steps(loader, max_step=10000, evaluate_every=50)
    
    # stop GAN training
    gan_mol.eval()

    # save the model weights
    torch.save(gan_mol.state_dict(), "initial_discr_mol_gan.pt")
    print('ok')
    
    steps = np.arange(len(initial_history["loss_disc"]))
    plt.plot(steps, initial_history["loss_disc"], label="Initial discriminator loss")
    plt.legend(loc="upper right")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()
    
    return gan_mol, initial_history
    

def discr_gen_train():
    # load data 
    data = pd.read_csv("concatenated_smiles.csv")
    x = data["smiles"]

    clf = pi.load(open("clf.pkl", "rb"))

    # load initially trained discriminator weights
    gan_mol = MolGen(x, clf, hidden_dim=64, lr=1e-4, device="cuda")
    gan_mol.load_state_dict(torch.load("initial_discr_mol_gan.pt"))

    # create dataloader
    loader = gan_mol.create_dataloader(x, batch_size=128, shuffle=True, num_workers=4)

    pretrain_history = gan_mol.train_n_steps(loader, max_step=5000, evaluate_every=50)
    
    # stop GAN training
    gan_mol.eval()
    print('ok')

    # stop model training and save the model weights
    torch.save(gan_mol.state_dict(), "pretrain_mol_gan.pt")
    
    # After training
    smiles_list = gan_mol.generate_n(100)

    valid_smiles = []
    for mol in smiles_list:
        if Chem.MolFromSmiles(mol) is not None:
            valid_smiles.append(Chem.MolFromSmiles(mol))
    
    Draw.MolsToGridImage(valid_smiles, molsPerRow=5)
    
    steps = np.arange(len(pretrain_history["loss_disc"]))
    plt.plot(steps, pretrain_history["loss_disc"], label="discriminator loss")
    plt.plot(steps, pretrain_history["loss_gen"], label="generator loss")
    plt.legend(loc="upper right")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()
    
    return gan_mol, pretrain_history

def second_train():
    # load data 
    data = pd.read_csv("concatenated_smiles.csv")
    x = data["smiles"]

    clf = pi.load(open("clf.pkl", "rb"))

    # load initially trained discriminator weights
    gan_mol = MolGen(x, clf, hidden_dim=64, lr=1e-4, device="cuda")
    gan_mol.load_state_dict(torch.load("pretrain_mol_gan.pt"))
    
    # set GAN to the training mode
    gan_mol.train()

    loader = gan_mol.create_dataloader(x, batch_size=128, shuffle=True, num_workers=4)

    coformer_history = gan_mol.train_n_steps_coformer(loader, max_step=5000, evaluate_every=50)
    
    gan_mol.eval()
    
    # save coformer trained GAN
    torch.save(gan_mol.state_dict(), "secondly_trained_gan_mol.pt")
    print("ok")
    
    # After training
    smiles_list = gan_mol.generate_n(100)

    valid_smiles = []
    for mol in smiles_list:
        if Chem.MolFromSmiles(mol) is not None:
            valid_smiles.append(Chem.MolFromSmiles(mol))
    
    Draw.MolsToGridImage(valid_smiles, molsPerRow=5)
    
    steps = len(coformer_history["loss_discr"])
    plt.plot(steps, coformer_history["loss_discr"], label="discriminator loss")
    plt.plot(steps, coformer_history["loss_gen"], label="generator loss")
    plt.legend(loc="upper right")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()
    
    return gan_mol, coformer_history
    