import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch import nn
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
import tensorflow as tf
import pickle as pi
import pandas as pd
from layers import Generator, RecurrentDiscriminator
from tokenizer import Tokenizer

RDLogger.DisableLog('rdApp.*')


# Nina
# clf = pi.load(open('clf.pkl', 'rb'))
# clf_plt = []
#
# drug_smiles = input()
# mol = Chem.MolFromSmiles(drug_smiles)
# drug = Chem.MolToSmiles(mol, kekuleSmiles=True)
# Nina


class MolGen(nn.Module):

    def __init__(self, data, clf_path, hidden_dim=128, lr=1e-3, device='cpu'):
        """[summary]

        Args:
            data (list[str]): [description]
            clf_path (str): path to the pretrained molecule classifier
            hidden_dim (int, optional): [description]. Defaults to 128.
            lr ([type], optional): learning rate. Defaults to 1e-3.
            device (str, optional): 'cuda' or 'cpu'. Defaults to 'cpu'.
        """
        super().__init__()

        self.device = device

        self.hidden_dim = hidden_dim

        self.tokenizer = Tokenizer(data)

        self.generator = Generator(
            latent_dim=hidden_dim,
            vocab_size=self.tokenizer.vocab_size - 1,
            start_token=self.tokenizer.start_token - 1,  # no need token
            end_token=self.tokenizer.end_token - 1,
        ).to(device)

        self.discriminator = RecurrentDiscriminator(
            hidden_size=hidden_dim,
            vocab_size=self.tokenizer.vocab_size,
            start_token=self.tokenizer.start_token,
            bidirectional=True
        ).to(device)

        self.generator_optim = torch.optim.Adam(
            self.generator.parameters(), lr=lr)

        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr)

        self.classifier = pi.load(open(clf_path, 'rb'))

        self.b = 0.  # baseline reward

    def load(self, path_to_model: str):
        """Loads the pre-trained model from the path

        :param path_to_model: pickle file containing weights of the pre-trained model
        :return: MolGen class pre-trained model instance
        """
        pretrained_model = torch.load(path_to_model)
        self.load_state_dict(pretrained_model)

    def save(self, save_filepath):
        """Saves the current model weights version to pickle (.pt/.pkl) format

        :param save_filepath: filepath for weights to be stored
        :return: None
        """
        torch.save(obj=self.state_dict(), f=save_filepath)

    def sample_latent(self, batch_size):
        """Sample from latent space

        Args:
            batch_size (int): number of samples

        Returns:
            torch.Tensor: [batch_size, self.hidden_dim]
        """
        return torch.randn(batch_size, self.hidden_dim).to(self.device)

    def discriminator_loss(self, x, y):
        """Discriminator loss

        Args:
            x (torch.LongTensor): input sequence [batch_size, max_len]
            y (torch.LongTensor): sequence label (zeros from generatoe, ones from real data)
                                  [batch_size, max_len]

        Returns:
            loss value
        """

        y_pred, mask = self.discriminator(x).values()

        loss = F.binary_cross_entropy(
            y_pred, y, reduction='none') * mask

        loss = loss.sum() / mask.sum()

        return loss

    def classifier_loss(self, smiles_batch):
        """
        :param smiles_batch: smiles array to be evaluated
        :return: classificator loss (cross-entropy between clf.predict_proba and desired properties)
        """

        batch_size = smiles_batch.shape[0]
        desired_value = 1

        descriptor_names = rdMolDescriptors.Properties.GetAvailableProperties()
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)

        properties_array = None
        for smiles in smiles_batch:
            mol_obj = Chem.MolFromSmiles(smiles)
            descriptors = get_descriptors.ComputeProperties(mol_obj)
            descriptors_array = np.array(descriptors)
            if properties_array is None:
                properties_array = np.array([descriptors_array])
            else:
                properties_array = np.append(properties_array, descriptors_array)

        properties_ds = pd.DataFrame(properties_array, columns=descriptor_names)

        clf_predict = self.classifier.predict_proba_(properties_ds)[:, desired_value]  # changed to predict_proba_
        clf_loss = tf.keras.metrics.binary_crossentropy(desired_value,
                                                        clf_predict)

        return clf_loss

    def train_step(self, x):
        """One training step

            Args:
                x (torch.LongTensor): sample form real distribution
            """

        batch_size, len_real = x.size()

        # create real and fake labels
        x_real = x.to(self.device)
        y_real = torch.ones(batch_size, len_real).to(self.device)

        # sample latent var
        z = self.sample_latent(batch_size)
        generator_outputs = self.generator.forward(z, max_len=100)
        x_gen, log_probs, entropies = generator_outputs.values()

        # label for fake data
        _, len_gen = x_gen.size()
        y_gen = torch.zeros(batch_size, len_gen).to(self.device)

        #####################
        # Train Discriminator
        #####################

        self.discriminator_optim.zero_grad()

        # disc fake loss
        fake_loss = self.discriminator_loss(x_gen, y_gen)

        # disc real loss
        real_loss = self.discriminator_loss(x_real, y_real)

        # combined disc loss
        discr_loss = 0.5 * (real_loss + fake_loss)

        # this is from paper, we change the overall loss function by adding classifier loss
        # discr_loss.backward()

        # classifier loss
        clf_loss = self.classifier_loss(x_gen)

        # combined disc + clf loss
        # combined_loss = np.log(discr_loss) + np.log(clf_loss)  # version 0

        combined_loss = np.log(np.log(discr_loss) + np.log(clf_loss)) + \
                        np.log(1 - (np.log(discr_loss) + np.log(clf_loss)))  # version 1

        combined_loss.backward()

        # clip grad
        clip_grad_value_(self.discriminator.parameters(), 0.1)

        # update params
        self.discriminator_optim.step()

        # ###############
        # Train Generator
        # ###############

        self.generator_optim.zero_grad()

        # prediction for generated x
        y_pred, y_pred_mask = self.discriminator(x_gen).values()

        # Reward (see the ref paper)
        R = (2 * y_pred - 1)

        # reward len for each sequence
        lengths = y_pred_mask.sum(1).long()

        # list of rew of each sequences
        list_rewards = [rw[:ln] for rw, ln in zip(R, lengths)]

        # compute - (r - b) log x
        generator_loss = []
        for reward, log_p in zip(list_rewards, log_probs):
            # substract the baseline
            reward_baseline = reward - self.b

            generator_loss.append((- reward_baseline * log_p).sum())

        # baseline moving average
        with torch.no_grad():
            mean_reward = (R * y_pred_mask).sum() / y_pred_mask.sum()
            self.b = 0.9 * self.b + (1 - 0.9) * mean_reward

        generator_loss.backward()

        clip_grad_value_(self.generator.parameters(), 0.1)

        self.generator_optim.step()

        return {'combined_loss': combined_loss.item(),
                'discr_loss': discr_loss.item(),
                'generator_loss': generator_loss.item(),
                'mean_reward': mean_reward}

    def create_dataloader(self, data, batch_size=128, shuffle=True, num_workers=5):
        """create a dataloader

        Args:
            data (list[str]): list of molecule smiles
            batch_size (int, optional): Defaults to 128.
            shuffle (bool, optional): Defaults to True.
            num_workers (int, optional): Defaults to 5.

        Returns:
            torch.data.DataLoader: a torch dataloader
        """

        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.tokenizer.batch_tokenize,
            num_workers=num_workers
        )

    def train_n_steps(self, train_loader, max_step=10000, evaluate_every=50):
        """Train for max_step steps

        Args:
            train_loader (torch.data.DataLoader): dataloader
            max_step (int, optional): Defaults to 10000.
            evaluate_every (int, optional): Defaults to 50.
        """

        iter_loader = iter(train_loader)

        history = {
            'discriminator_loss': np.array([]),
            'generator_loss': np.array([]),
            'classifier_loss': np.array([])
        }

        for step in range(max_step):

            try:
                batch = next(iter_loader)
            except:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            # model update
            self.train_step(batch)

            if step % evaluate_every == 0:
                self.eval()
                score = self.evaluate_n(100)
                self.train()

                history['discriminator_loss'] = np.append(history['discriminator_loss'],
                                                          score)

                # Nina
                # clf_plt.append(score[1])
                #
                # # if score > best_score:
                # #     self.save_best()
                # #     print('saving')
                # #     best_score = score
                #
                # print(f'valid = {score[0]: .2f}')
                # print(f'clf loss = {score[1]: .2f}') # Nina
                # print(clf_plt)

    def get_mapped(self, seq):
        """Transform a sequence of ids to string

        Args:
            seq (list[int]): sequence of ids

        Returns:
            str: string output
        """
        return ''.join([self.tokenizer.inv_mapping[i] for i in seq])

    @torch.no_grad()
    def generate_n(self, n):
        """Generate n molecules

        Args:
            n (int)

        Returns:
            list[str]: generated molecules
        """

        z = torch.randn((n, self.hidden_dim)).to(self.device)

        x = self.generator(z)['x'].cpu()

        lenghts = (x > 0).sum(1)

        # l - 1 because we exclude end tokens
        return [self.get_mapped(x[:l - 1].numpy()) for x, l in zip(x, lenghts)]

    def evaluate_n(self, n):
        """Evaluation: frequence of valid molecules using rdkit

        Args:
            n (int): number of sample

        Returns:
            float: requence of valid molecules
        """
        pack = self.generate_n(n)
        print(pack[:2])

        valid = np.array([Chem.MolFromSmiles(k) is not None for k in pack])
        clf_loss = self.classifier_loss(pack)  # Nina

        return valid.mean(), clf_loss.mean()
