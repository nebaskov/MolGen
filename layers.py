import torch
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders import (LstmSeq2SeqEncoder,
                                               PytorchTransformer)
from torch import nn
from torch.distributions import Categorical
from torch.nn.modules.activation import Sigmoid
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, latent_dim, vocab_size, start_token, end_token):
        """Generator

        Args:
            latent_dim (int): [description]
            vocab_size (int): vocab size without padding
            start_token ([int]): start token (without padding idx)
            end_token ([int]): end token (without padding idx)
        """

        super().__init__()

        # (-1) we do not need pad token for the generator
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token

        self.embedding_layer = nn.Embedding(self.vocab_size, latent_dim)

        self.project = FeedForward(
            input_dim=latent_dim,
            num_layers=2,
            hidden_dims=[latent_dim * 2, latent_dim * 2],
            activations=[nn.ReLU(), nn.ELU(alpha=0.1)],
            dropout=[0.1, 0.3]
        )

        self.rnn = nn.LSTMCell(latent_dim, latent_dim)

        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim * 2, vocab_size - 1)
        )

    def forward(self, z, max_len=50):
        """[summary]

        Args:
            z (torch.Tensor): [description]
            max_len (int, optional): [description]. Defaults to 20.

        Returns:
            dict: x [B, max_len], log_probabilities [B, max_len, vocab], entropies [B,]
        """

        batch_size = z.shape[0]

        # start of sequence
        starts = torch.full(
            size=(batch_size,), fill_value=self.start_token, device=z.device).long()

        # embed_start
        emb = self.embedding_layer(starts)

        x = []
        log_probabilities = []
        entropies = []

        h, c = self.project(z).chunk(2, dim=1)

        for i in range(max_len):

            # new state
            h, c = self.rnn(emb, (h, c))

            # prediction
            logits = self.output_layer(h)

            # create dist
            dist = Categorical(logits=logits)

            # sample
            sample = dist.sample()

            # append prediction
            x.append(sample)

            # append log prob
            log_probabilities.append(dist.log_prob(sample))

            # append entropy
            entropies.append(dist.entropy())

            # new embedding
            emb = self.embedding_layer(sample)

        # stack along sequence dim
        x = torch.stack(x, dim=1)
        log_probabilities = torch.stack(log_probabilities, dim=1)
        entropies = torch.stack(entropies, dim=1)

        # keep only valid lengths (before EOS)
        end_pos = (x == self.end_token).float().argmax(dim=1).cpu()

        # sequence length is end token position + 1
        seq_lengths = end_pos + 1

        # if end_pos = 0 => put seq_length = max_len
        seq_lengths.masked_fill_(seq_lengths == 1, max_len)

        # select up to length
        _x = []
        _log_probabilities = []
        _entropies = []
        for x_i, logp, ent, length in zip(x, log_probabilities, entropies, seq_lengths):
            _x.append(x_i[:length])
            _log_probabilities.append(logp[:length])
            _entropies.append(ent[:length].mean())

        x = torch.nn.utils.rnn.pad_sequence(
            _x, batch_first=True, padding_value=-1)

        x = x + 1  # add padding token

        return {'x': x, 'log_probabilities': _log_probabilities, 'entropies': _entropies}


class RecurrentDiscriminator(nn.Module):

    def __init__(self, hidden_size, vocab_size, start_token, bidirectional=False):
        """Reccurent discriminator

        Args:
            hidden_size (int): model hidden size
            vocab_size (int): vocabulary size
            bidirectional (bool, optional): [description]. Defaults to True.
        """

        super().__init__()

        self.start_token = start_token

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        self.rnn = LstmSeq2SeqEncoder(
            hidden_size, hidden_size, num_layers=1, bidirectional=bidirectional)

        if bidirectional:
            hidden_size = hidden_size * 2

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """

        batch_size, _ = x.size()

        # append start token to the input
        starts = torch.full(
            size=(batch_size, 1), fill_value=self.start_token, device=x.device).long()

        x = torch.cat([starts, x], dim=1)

        mask = x > 0

        # embed input [batch_size, max_len, hidden_size]
        emb = self.embedding(x)

        # contextualize representation
        x = self.rnn(emb, mask)

        # prediction for each sequence
        out = self.fc(x).squeeze(-1)  # [B, max_len]

        return {'out': out[:, 1:], 'mask': mask.float()[:, 1:]}


class JSD(nn.Module):
    """Jensen Shannon Divergence loss"""

    def __init__(self, device):
        super(JSD, self).__init__()
        self.device = torch.device(device)
    
    def forward(self, real_logs, gen_logs):
        
        batch_size = real_logs.size()[0]
        
        # cut_real_logs = torch.tensor([])
        # for tensor in real_logs:
        #     stop_idx = 0    
        #     for element in tensor.values():
        #         if element == 0:
        #             stop_idx += 1
        #             break
        #         else:
        #             stop_idx += 1
            
            
        # real_probs =  F.softmax(real_logs, dim=1, dtype=torch.float16)
        # gen_probs =  F.softmax(gen_logs, dim=1, dtype=torch.float16)
        
        real_dist = Categorical(logits=real_logs)
        real_sample = real_dist.sample()
        real_probs = torch.exp(real_dist.log_prob(real_sample)).to(self.device)
        
        gen_dist = Categorical(logits=gen_logs)
        gen_sample = gen_dist.sample()
        gen_probs = torch.exp(gen_dist.log_prob(gen_sample)).to(self.device)
        
        
        real_len = real_probs.size()[0]
        gen_len = gen_probs.size()[0]

        if real_len > gen_len:
            diff = real_len - gen_len
            zeros = torch.zeros((batch_size, diff), device=self.device, dtype=torch.float16)
            gen_probs = torch.cat([gen_probs, zeros], dim=1, dtype=torch.float16)
            gen_logits = torch.cat([gen_logits, zeros], dim=1, dtype=torch.float16)
            
        elif real_len < gen_len:
            diff = gen_len - real_len
            zeros = torch.zeros((batch_size, diff), device=self.device, dtype=torch.float16)
            real_probs = torch.cat([real_probs, zeros], dim=1, dtype=torch.float16)
            real_logits = torch.cat([real_logits, zeros], dim=1, dtype=torch.float16)
        
        total_m = (0.5 * (real_probs.to(self.device) +
                          gen_probs.to(self.device))).to(self.device)
                          
        loss = 0.0
        loss += F.kl_div(real_probs, total_m, reduction="batchmean") 
        loss += F.kl_div(gen_probs, total_m, reduction="batchmean") 
        loss *= 0.5
        
        # loss += F.kl_div(F.log_softmax(real_logits, dim=1), total_m, reduction="batchmean") 
        # loss += F.kl_div(F.log_softmax(gen_logits, dim=1), total_m, reduction="batchmean") 
        # loss *= 0.5

        loss.to(self.device)
        
        return loss
    
    
# class JSD(nn.Module):
#     def __init__(self):
#         super(JSD, self).__init__()
#         self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

#     def forward(self, p: torch.tensor, q: torch.tensor):
#         p, q = F.softmax(p, dim=0), F.softmax(q, dim=0)
#         p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
#         m = (0.5 * (p + q)).log()
        
#         return 0.5 * (self.kl(p.log(), m) + self.kl(q.log(), m))