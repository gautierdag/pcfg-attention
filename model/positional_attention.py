import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalAttention(nn.Module):
    def __init__(self,
                 feature_dim,
                 positioning_embedding=20,
                 num_building_blocks=3):
        super(PositionalAttention, self).__init__()
        self.num_building_blocks = num_building_blocks

        self.positioning_generator = nn.LSTM(
            feature_dim, positioning_embedding, batch_first=True)

        self.sigma_generator = nn.Linear(positioning_embedding, 1)
        self.mu_generator = nn.Linear(
            positioning_embedding, num_building_blocks)

        self.prev_mu = None

    @staticmethod
    def normal_pdf(x, mu, sigma):
        """Return normalized Gaussian_pdf(x)."""
        x = torch.exp(-(x - mu)**2 / (2 * sigma**2 + 10e-4))
        # Normalize the Gaussian PDF result
        x = F.normalize(x, p=1)
        return x

    def forward(self, decoder_states, encoder_states, **kwargs):
        # Input x is encoder output return_attention decides whether to return
        # attention scores over the encoder output

        # compute mask
        mask = encoder_states.eq(0.)[:, :, :1]

        # decoder_states --> (batch, dec_seqlen, hl_size)
        batch_size, enc_seqlen, hl_size = encoder_states.size()

        # POSITIONAL ATTENTION

        # Need the lengths to normalize each sentence to respective length
        # for the building blocks - 1/N and j/N
        sentence_lengths = mask.transpose(1, 2).argmax(
            dim=2).flatten().type(torch.FloatTensor)
        sentence_lengths[sentence_lengths == 0] = enc_seqlen

        positioning_weights, _ = self.positioning_generator(
            decoder_states[:, -1, :].unsqueeze(1))

        mu_weights = F.relu(self.mu_generator(positioning_weights))
        sigma = torch.sigmoid(
            self.sigma_generator(positioning_weights)).squeeze()

        # Setting up Building Blocks
        if kwargs['unroll_step'] == 0:
            self.prev_mu = torch.zeros(batch_size, device=device)

        building_blocks = torch.ones(
            (batch_size, self.num_building_blocks), device=device)

        building_blocks[:, 0] = self.prev_mu
        building_blocks[:, 1] = 1 / sentence_lengths
        building_blocks[:, 2] = (kwargs['unroll_step']+1) / sentence_lengths

        mu = torch.bmm(mu_weights, building_blocks.unsqueeze(2)
                       ).squeeze()

        # need to clamp to direct attention to within sequence
        # @TODO: Not sure clamping makes sense here
        mu = torch.min(mu, torch.ones(batch_size, device=device))
        self.prev_mu = mu

        # relative counter that represents 0-1 where to attend on sequence up until now
        rel_counter = torch.arange(
            enc_seqlen, dtype=torch.float, device=device)
        rel_counter = rel_counter.expand(
            batch_size, -1) / sentence_lengths.view(batch_size, 1)

        attn = self.normal_pdf(
            rel_counter, mu.unsqueeze(1), sigma.unsqueeze(1)).unsqueeze(2)

        # apply local mask
        attn.masked_fill_(mask, -float('inf'))

        attn = F.softmax(attn.view(-1, enc_seqlen),
                         dim=1).view(batch_size, -1, enc_seqlen)

        context = torch.bmm(attn, encoder_states)

        return context, attn
