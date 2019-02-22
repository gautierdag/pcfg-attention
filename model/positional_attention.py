import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class Dot(nn.Module):

#     def __init__(self):
#         super(Dot, self).__init__()

#     def forward(self, decoder_states, encoder_states):
#         attn = torch.bmm(decoder_states, encoder_states.transpose(1, 2))
#         return attn


# class MLP(nn.Module):

#     def __init__(self, dim):
#         super(MLP, self).__init__()
#         self.mlp = nn.Linear(dim * 2, dim)
#         self.activation = nn.ReLU()
#         self.out = nn.Linear(dim, 1)

#     def forward(self, decoder_states, encoder_states):


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

    @staticmethod
    def normal_pdf(x, mu, sigma):
        """Return normalized Gaussian_pdf(x)."""
        x = torch.exp(-(x - mu)**2 / (2 * sigma**2 + 10e-4))
        # Normalize the Gaussian PDF result
        x = F.normalize(x, p=1)
        return x

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, decoder_states, encoder_states,
                **attention_method_kwargs):

        # Input x is encoder output
        # return_attention decides whether to return
        # attention scores over the encoder output

        batch_size = decoder_states.size(0)
        decoder_states.size(2)
        sequence_length = encoder_states.size(1)

        # compute mask
        mask = encoder_states.eq(0.)[:, :, :1].transpose(1, 2)

        # COMPUTE ATTENTION VALUES
        # apply mlp to all encoder states for current decoder

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, hl_size = encoder_states.size()
        _, dec_seqlen, _ = decoder_states.size()

        # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        encoder_states_exp = encoder_states.unsqueeze(1)
        encoder_states_exp = encoder_states_exp.expand(
            batch_size, dec_seqlen, enc_seqlen, hl_size)

        # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        decoder_states_exp = decoder_states.unsqueeze(2)
        decoder_states_exp = decoder_states_exp.expand(
            batch_size, dec_seqlen, enc_seqlen, hl_size)

        # reshape encoder and decoder states to allow batchwise computation. We will have
        # batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear
        # layer for each of them
        decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size)
        encoder_states_tr = encoder_states_exp.contiguous().view(-1, hl_size)

        mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)

        # apply mlp and reshape to get in correct form
        mlp_output = self.mlp(mlp_input)
        mlp_output = self.activation(mlp_output)
        out = self.out(mlp_output)
        attn = out.view(batch_size, dec_seqlen, enc_seqlen)

        if self.mask is not None:
            attn.masked_fill_(self.mask, -float('inf'))

        # apply local mask
        attn.masked_fill_(mask, -float('inf'))

        attn = F.softmax(attn.view(-1, sequence_length),
                         dim=1).view(batch_size, -1, sequence_length)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        context = torch.bmm(attn, encoder_states)

        return context, attn

        # # Need the lengths to normalize each sentence to respective length
        # # for the building blocks - 1/N and j/N
        # sentence_lengths = pad_lengths.expand(
        #     sequence_length, batch_size)

        # # pack for efficiency if more than one element (else unpadded)
        # if not return_attention:
        #     packed_input = pack_padded_sequence(x, pad_lengths,
        #                                         batch_first=True)
        #     packed_output, _ = self.positioning_generator(packed_input)
        #     positioning_weights, _ = pad_packed_sequence(packed_output, batch_first=True,
        #                                                  total_length=sequence_length)
        # else:
        #     positioning_weights, _ = self.positioning_generator(x)

        # mu_weights = F.relu(self.mu_generator(positioning_weights))
        # sigma_weights = torch.sigmoid(
        #     self.sigma_generator(positioning_weights))

        # # Setting up Building Blocks
        # prev_mu = torch.zeros(batch_size, device=device)
        # building_blocks = torch.ones(
        #     (sequence_length, batch_size, self.num_building_blocks), device=device)
        # building_blocks[:, :, 1] = 1/sentence_lengths
        # building_blocks[:, :, 2] = (torch.arange(
        #     sequence_length, dtype=torch.float, device=device)+1).unsqueeze(1).expand(-1, batch_size) / sentence_lengths

        # # Attend for each time step using the previous context
        # position_vectors = []  # Which positions to attend to
        # attention_vectors = []

        # # we go over the whole sequence - even though it is padded so the max
        # # length might be shorter.
        # for j in range(sequence_length):
        #     # For each timestep the context that is attented grows
        #     # as there are more available previous hidden states
        #     bb = building_blocks[j].clone()
        #     bb[:, 0] = prev_mu

        #     mu = torch.bmm(mu_weights[:, j, :].clone(
        #     ).unsqueeze(1), bb.unsqueeze(2)).squeeze()

        #     # need to clamp to direct attention to previous segment of sequence
        #     # max dynamic and expands as we look further down sequence
        #     mu = torch.max(mu, j/pad_lengths)
        #     prev_mu = mu

        #     sigma = sigma_weights[:, j, :]

        #     # relative counter that represents 0-1 where to attend on sequence up till now
        #     rel_counter = torch.arange(
        #         j+1, dtype=torch.float, device=device)
        #     rel_counter = rel_counter.expand(
        #         batch_size, -1) / pad_lengths.view(batch_size, 1)

        #     gaussian_weighted_attention = self.normal_pdf(
        #         rel_counter, mu.unsqueeze(1), sigma).unsqueeze(2)

        #     # multiply the weights with the hidden encoded states found till this point
        #     applied_positional_attention = x[:, :j+1,
        #                                      :].clone() * gaussian_weighted_attention
        #     position_vectors.append(
        #         torch.sum(applied_positional_attention, dim=1))

        #     if return_attention:
        #         attention_vectors.append(
        #             gaussian_weighted_attention.cpu().detach().numpy())

        # context_vectors = torch.stack(position_vectors).transpose(0, 1)
