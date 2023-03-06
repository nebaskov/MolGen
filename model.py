import statistics

import numpy as np
import pickle as pi
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_
from torch.distributions import Categorical

from tokenizer import Tokenizer
from layers import Generator, RecurrentDiscriminator, JSD

RDLogger.DisableLog('rdApp.*')


class MolGen(nn.Module):

    def __init__(self, data, classifier, hidden_dim=128, lr=1e-3, device='cpu'):
        """[summary]

        Args:
            data (list[str]): [description]
            classifier: pretrained classifier model
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

        self.js_div = JSD(self.device).to(device)

        
        self.generator_optim = torch.optim.Adam(
            self.generator.parameters(), lr=lr)

        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr)

        self.b = 0.  # baseline reward
        
        self.classifier = classifier

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
            y (torch.LongTensor): sequence label (zeros from generator, ones from real data)
                                  [batch_size, max_len]

        Returns:
            loss value
        """

        y_pred, mask = self.discriminator(x).values()

        loss = F.binary_cross_entropy(
            y_pred, y, reduction='none') * mask

        loss = loss.sum() / mask.sum()

        return loss
    
    def initial_train(self, x):
        """Initial GAN trainig

        Args:
            x (torch.LongTensor): sample from real distribution
            
        Returns:
            float: discriminator loss
        """
        
        batch_size, len_real = x.size()

        # create real and fake labels
        x_real = x.to(self.device)
        y_real = torch.ones(batch_size, len_real).to(self.device)

        # sample latent var
        z = self.sample_latent(batch_size)
        generator_outputs = self.generator.forward(z, max_len=20)
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

        # combined loss
        discr_loss = 0.5 * (real_loss + fake_loss)
        discr_loss.backward()

        # clip grad
        clip_grad_value_(self.discriminator.parameters(), 0.1)

        # update params
        self.discriminator_optim.step()
        
        return discr_loss.item()
        
    def initial_train_n_steps(self, train_loader, max_step=10000, evaluate_every=50):
        """Train for max_step steps

        Args:
            train_loader (torch.data.DataLoader): dataloader
            max_step (int, optional): Defaults to 10000.
            evaluate_every (int, optional): Defaults to 50.
        
        Returns: 
            dict: "loss_disc": [values]
        """

        iter_loader = iter(train_loader)

        # best_score = 0.0
        history = {"loss_disc": []}
        for step in range(max_step):

            try:
                batch = next(iter_loader)
            except:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            # model update
            local_history = self.initial_train(batch)
            history["loss_disc"].append(local_history)
            
            
            # if step % evaluate_every == 0:

            #     self.eval()
            #     score = self.evaluate_n(100)
            #     history["overall_valid"].append(score)
            #     self.train()

            #     print(f"discriminator loss: {history['loss_disc']} \n \n")
                
        return history

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
        generator_outputs = self.generator.forward(z, max_len=20)
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

        # combined loss
        discr_loss = 0.5 * (real_loss + fake_loss)
        discr_loss.backward()

        # clip grad
        clip_grad_value_(self.discriminator.parameters(), 0.1)

        # update params
        self.discriminator_optim.step()

        # ###############
        # Train Generator
        # ###############

        self.generator_optim.zero_grad()

        # prediction for generated x
        y_pred, y_pred_mask = self.discriminator.forward(x_gen).values()

        # Reward (see the ref paper)
        R = (2 * y_pred - 1)

        # reward len for each sequence
        lengths = y_pred_mask.sum(1).long()

        # list of rewards of each sequences
        list_rewards = [rw[:ln] for rw, ln in zip(R, lengths)]

        # compute - (r - b) log x
        generator_loss = []
        for reward, log_p in zip(list_rewards, log_probs):

            # substract the baseline
            reward_baseline = reward - self.b

            generator_loss.append((- reward_baseline * log_p).sum() ** 2)

        # mean loss + entropy reg
        generator_loss = torch.stack(generator_loss).mean() - \
            sum(entropies) * 0.01 / batch_size
            
        generator_loss.backward()
        
        # real_dist = Categorical(logits=y_real)
        # real_sample = real_dist.sample()
        # real_log_proba = real_dist.log_prob(real_sample)
        # real_proba = torch.exp(real_log_proba)
        
        # fake_dist = Categorical(logits=y_gen)
        # fake_sample = fake_dist.sample()
        # fake_log_proba = fake_dist.log_prob(fake_sample)
        # fake_proba = torch.exp(fake_log_proba)
        
        # calculate Jensen-Shannon divergence
        # d_js = []
        # for r_prob, f_prob in zip(real_proba, fake_proba):
            
        #     local_d_js = f_prob * torch.log((2 * r_prob) / (r_prob + f_prob)) + \
        #         r_prob * torch.log((2 * r_prob / (r_prob + f_prob)))
            
        #     d_js.append(local_d_js)
        
        # test_loss = (self.js_div.forward(x_real, x_gen) + \
        #                 (generator_loss.to(self.device) * 0).to(torch.float16).to(self.device)).to(self.device)


        # test_loss.backward(retain_graph=False)

        
        # baseline moving average
        with torch.no_grad():
            mean_reward = (R * y_pred_mask).sum() / y_pred_mask.sum()
            self.b = 0.9 * self.b + (1 - 0.9) * mean_reward

        # generator_loss.backward()

        clip_grad_value_(self.generator.parameters(), 0.1)

        self.generator_optim.step()

        # return {'loss_disc': discr_loss.item(), 'mean_reward': mean_reward, "loss_gen": generator_loss.item()}
        return {'loss_disc': discr_loss.item(), 'mean_reward': mean_reward, "loss_gen": test_loss}

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

    def train_n_steps(self, train_loader, mode, max_step=10000, evaluate_every=50):
        """Train for max_step steps

        Args:
            train_loader (torch.data.DataLoader): dataloader
            mode (str): "init" or "pretrain" or "coformer" - iteration mode
            max_step (int, optional): Defaults to 10000.
            evaluate_every (int, optional): Defaults to 50.
        
        Returns: 
            dict("loss_disc": [values], "loss_gen": [values], "overall_valid": [values])
        """

        iter_loader = iter(train_loader)

        # best_score = 0.0
        history = {"loss_disc": [], "loss_gen": [], "overall_valid": []}
        for step in range(max_step):

            try:
                batch = next(iter_loader)
            except:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            # model update
            local_history = self.train_step(batch)
            
            # model save best
            # if step > 0:
            #     discr_loss_condition = statistics.mean(history["loss_disc"]) > statistics.mean(local_history["loss_disc"])
            #     gen_loss_condition = st.mean(history["gen_loss_disc"]) > statistics.mean(local_history["gen_loss_disc"])                
                
            #     if discr_loss_condition or gen_loss_condition: 
            #         torch.save(self.state_dict(), f"{mode}_best_model.pt")

                
            history["loss_disc"].append(local_history["loss_disc"])
            history["loss_gen"].append(local_history["loss_gen"])
            
            if step % evaluate_every == 0:

                self.eval()
                score = self.evaluate_n(100)
                history["overall_valid"].append(score)
                self.train()

                # if score > best_score:
                #     self.save_best()
                #     print('saving')
                #     best_score = score

                # print(f'valid = {score: .2f}')
                # print(f"model history: \n",
                #       f"discriminator loss: {history['loss_disc']} \n \n",
                #       f"generator loss: {history['loss_gen']} \n \n",
                #       f"valid: {history['overall_valid']} \n \n")
                print(f"valid: {history['overall_valid'][-1]} \n \n")
                
        return history

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
        return [self.get_mapped(x[:l-1].numpy()) for x, l in zip(x, lenghts)]

    def evaluate_n(self, n):
        """Evaluation: frequence of valid molecules using rdkit

        Args:
            n (int): number of sample

        Returns:
            float: requence of valid molecules
        """

        pack = self.generate_n(n)
        
        valid_pack = []
        
        for mol in pack:
            if Chem.MolFromSmiles(mol) is not None:
                valid_pack.append(mol)
        
        print(valid_pack)

        valid = np.array([Chem.MolFromSmiles(k) is not None for k in pack])

        return valid.mean()
    
    ###############################################
    # TRAIN GAN ON COFORMER SMILES AND SPECIFIC DRUG
    ###############################################
    
    @staticmethod
    def generate_descriptors(smiles):
  
        descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)
        
        molecule_object = Chem.MolFromSmiles(smiles)
        final_descriptors = np.array(get_descriptors.ComputeProperties(molecule_object)).reshape((-1, 43))

        return final_descriptors


    def get_clf_input(self, drug_smiles, coformer_smiles):
        drug_descriptors, coformer_descriptors = self.generate_descriptors(drug_smiles), self.generate_descriptors(coformer_smiles)

        final_input = np.concatenate((drug_descriptors, coformer_descriptors), axis=1)

        return final_input

    def calculate_clf_error(self, drug_smiles, coformer_smiles, desired_clf_output=1):

        clf_input = self.get_clf_input(drug_smiles, coformer_smiles)
        clf_prediction = self.classifier.predict_proba(clf_input)[:,desired_clf_output]
        clf_pred_tensor = torch.asarray(clf_prediction, dtype=torch.double)
        
        desired_target = torch.asarray([desired_clf_output], dtype=torch.double)
        error = F.binary_cross_entropy(input=clf_pred_tensor, target=desired_target)

        return error


    def train_step_coformer(self, x, drug_smiles: str):
        """One training step

        Args:
            x (torch.LongTensor): sample form real distribution
            drug_smiles (str): target molecule smiles
        """

        batch_size, len_real = x.size()

        # create real and fake labels
        x_real = x.to(self.device)
        y_real = torch.ones(batch_size, len_real).to(self.device)

        # sample latent var
        z = self.sample_latent(batch_size)
        generator_outputs = self.generator.forward(z, max_len=20)
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
        discr_loss = 0.5 * (real_loss + fake_loss)
        
        # classifier loss
        score, clf_loss = self.evaluate_n_coformer(batch_size, drug_smiles)
       
        # combined loss - it doesn't work because of negative numbers in torch.log()
        # log_discr_loss = torch.log(discr_loss)
        # log_clf_loss = torch.log(clf_loss.mean())
        # log_discr_clf_sum_loss = log_discr_loss + log_clf_loss
        
        # combined_loss = torch.log(log_discr_clf_sum_loss) + torch.log(1 - log_discr_clf_sum_loss)  # version 1

        combined_loss = (discr_loss + clf_loss.mean()) / 2

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

        # mean loss + entropy reg
        generator_loss = torch.stack(generator_loss).mean() - \
            sum(entropies) * 0.01 / batch_size

        # baseline moving average
        with torch.no_grad():
            mean_reward = (R * y_pred_mask).sum() / y_pred_mask.sum()
            self.b = 0.9 * self.b + (1 - 0.9) * mean_reward

        generator_loss.backward()

        clip_grad_value_(self.generator.parameters(), 0.1)

        self.generator_optim.step()

        return {'loss_disc': discr_loss.item(), 'mean_reward': mean_reward,
                "loss_gen": generator_loss.item(), "loss_clf": clf_loss.mean().item()}
    
    def train_n_steps_coformer(self, train_loader, drug_smiles: str, max_step=10000, evaluate_every=50):
        """Train for max_step steps

        Args:
            train_loader (torch.data.DataLoader): dataloader
            drug_smiles (str): target molecule smiles
            max_step (int, optional): Defaults to 10000.
            evaluate_every (int, optional): Defaults to 50.
        
        Returns: 
            dict("loss_disc": [values], "loss_gen": [values], "loss_clf": [values], "overall_valid": [values])
        """

        iter_loader = iter(train_loader)

        # best_score = 0.0
        history = {"loss_disc": [], "loss_gen": [], "loss_clf": [], "overall_valid": []}
        for step in range(max_step):

            try:
                batch = next(iter_loader)
            except:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            # model update
            local_history = self.train_step_coformer(batch, drug_smiles)
            history["loss_disc"].append(local_history["loss_disc"])
            history["loss_gen"].append(local_history["loss_gen"])
            history["loss_clf"].append(local_history["loss_clf"])
            
            if step % evaluate_every == 0:

                self.eval()
                score, clf_loss = self.evaluate_n_coformer(batch.size()[0], drug_smiles)
                history["overall_valid"].append(score)
                self.train()

                # if score > best_score:
                #     self.save_best()
                #     print('saving')
                #     best_score = score

                # print(f'valid = {score: .2f}')
                print(f"model history: \n",
                      f"discriminator loss: {history['loss_disc']} \n \n",
                      f"generator loss: {history['loss_gen']} \n \n",
                      f"classifier loss: {history['loss_clf']} \n \n",
                      f"valid: {history['overall_valid'][-1]} \n \n")
                
        return history
    
    def evaluate_n_coformer(self, n: int, drug_smiles: str):
        """Evaluation: frequence of valid molecules using rdkit

        Args:
            n (int): number of sample
            drug_smiles (str): target molecule smiles
            
        Returns:
            float: requence of valid molecules
            numpy.array: classifier loss on generated smiles
        """

        pack = self.generate_n(n)
        valid_pack = []
        
        for mol in pack:
            if Chem.MolFromSmiles(mol) is not None:
                valid_pack.append(mol)
        
        print(valid_pack)
                
        clf_loss = torch.tensor([])
        for idx in range(n):
            if Chem.MolFromSmiles(pack[idx]) is None:
                local_clf_loss = torch.tensor([1.0])
            else:
                local_clf_loss = self.calculate_clf_error(pack[idx], drug_smiles)
                local_clf_loss = torch.tensor([local_clf_loss.item()])
                
            clf_loss = torch.concat([clf_loss, local_clf_loss])
            
        valid = np.array([Chem.MolFromSmiles(k) is not None for k in pack])

        return valid.mean(), clf_loss