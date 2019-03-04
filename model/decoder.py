import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from machine.models.baseRNN import BaseRNN

# Custom implementations of Positional and standard attentions
from .positional_attention import PositionalAttention
from .attention import Attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoderRNN(BaseRNN):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)
        attention_method('str', optional): Method used in attention mechanism
            Required with use_attention, str from options {'mlp', 'dot', 'concat'}
        use_positional_attention(bool, optional): flag indication whether to
            use positional attention mechanism or not (default: false) (is always used post-rnn)
        full_focus(bool, optional): flag indication whether to use full attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_POS_ATTN_SCORE (str): key used to indicate positional attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_POS_ATTN_SCORE = 'position_attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0,
                 use_attention=False, attention_method=None, full_focus=False,
                 use_positional_attention=False, positioning_generator_size=20,
                 attention_mixer='sum'):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p,
                                         n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        input_size = hidden_size

        if (use_attention and attention_method is None):
            raise ValueError(
                "Method for computing attention should be provided")

        self.attention_method = attention_method
        self.full_focus = full_focus

        # increase input size decoder if attention is applied before decoder
        # rnn
        if use_attention == 'pre-rnn' and not full_focus:
            input_size *= 2

        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers,
                                 batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.use_positional_attention = use_positional_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size, self.attention_method)
        else:
            self.attention = None

        if use_positional_attention:
            self.positional_attention = PositionalAttention(
                self.hidden_size, positioning_embedding=positioning_generator_size)
        else:
            self.positional_attention = None

        if use_attention == 'post-rnn' or use_positional_attention:
            self.out = nn.Linear(2 * self.hidden_size, self.output_size)
        else:
            self.out = nn.Linear(self.hidden_size, self.output_size)
            if self.full_focus:
                self.ffocus_merge = nn.Linear(
                    2 * self.hidden_size, hidden_size)

        # When both attentions are applied at same level - we mix them
        self.mix_attention = (
            use_attention == 'post-rnn' and use_positional_attention)
        self.attention_mixer = attention_mixer
        if self.attention_mixer == 'mixer':
            # initialize alpha to 0.5
            self.alpha = torch.nn.Parameter(torch.ones(1, device=device)*0.5)
            self.alpha.requires_grad = True

    def forward_step(self, input_var, hidden,
                     encoder_outputs, function, **kwargs):
        """
        Performs one or multiple forward decoder steps.

        Args:
            input_var (torch.tensor): Variable containing the input(s) to the decoder RNN
            hidden (torch.tensor): Variable containing the previous decoder hidden state.
            encoder_outputs (torch.tensor): Variable containing the target outputs of the decoder RNN
            function (torch.tensor): Activation function over the last output of the decoder RNN at every time step.

        Returns:
            predicted_softmax: The output softmax distribution at every time step of the decoder RNN
            hidden: The hidden state at every time step of the decoder RNN
            attn: The attention distribution at every time step of the decoder RNN
        """
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        attn = None
        pos_attn = None

        if self.use_attention == 'pre-rnn':
            h = hidden
            if isinstance(hidden, tuple):
                h, _ = hidden
            # Apply the attention method to get the attention vector and weighted context vector.
            # Provide decoder step for hard attention transpose to get batch at the second index
            context, attn = self.attention(
                h[-1:].transpose(0, 1), encoder_outputs)
            combined_input = torch.cat((context, embedded), dim=2)
            if self.full_focus:
                merged_input = F.relu(self.ffocus_merge(combined_input))
                combined_input = torch.mul(context, merged_input)

            output, hidden = self.rnn(combined_input, hidden)

        # if mixing attention
        elif self.mix_attention:
            output, hidden = self.rnn(embedded, hidden)
            # Apply the attention method to get the attention vector and
            # weighted context vector. Provide decoder step for hard attention
            context, attn = self.attention(output, encoder_outputs)
            positional_context, pos_attn = self.positional_attention(
                output, encoder_outputs, **kwargs)

            # mix by summing
            if self.attention_mixer == 'sum':
                context = context+positional_context
            elif self.attention_mixer == 'mean':
                context = (context+positional_context)*0.5
            elif self.attention_mixer == 'mixer':
                context = self.alpha*context + \
                    (1-self.alpha)*positional_context
            else:
                raise ValueError(
                    'Attention mixer {} not implemented'.format(self.attention_mixer))

            output = torch.cat((context, output), dim=2)

        elif self.use_attention:
            output, hidden = self.rnn(embedded, hidden)
            # Apply the attention method to get the attention vector and
            # weighted context vector. Provide decoder step for hard attention
            context, attn = self.attention(output, encoder_outputs)
            output = torch.cat((context, output), dim=2)

        elif self.use_positional_attention:
            output, hidden = self.rnn(embedded, hidden)
            # Apply the attention method to get the attention vector and
            # weighted context vector. Provide decoder step for hard attention
            context, pos_attn = self.positional_attention(
                output, encoder_outputs, **kwargs)
            output = torch.cat((context, output), dim=2)

        elif not self.use_attention and not self.use_positional_attention:
            output, hidden = self.rnn(embedded, hidden)

        predicted_softmax = function(self.out(
            output.contiguous().view(-1, self.out.in_features)), dim=1).view(batch_size, output_size, -1)

        return predicted_softmax, hidden, attn, pos_attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0):

        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()
        if self.use_positional_attention:
            ret_dict[DecoderRNN.KEY_POS_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)

        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn, pos_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            if self.use_positional_attention:
                ret_dict[DecoderRNN.KEY_POS_ATTN_SCORE].append(pos_attn)

            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # When we use pre-rnn attention we must unroll the decoder. We need to calculate the attention based on
        # the previous hidden state, before we can calculate the next hidden state.
        # We also need to unroll when we don't use teacher forcing. We need perform the decoder steps
        # one-by-one since the output needs to be copied to the input of the
        # next step.
        if self.use_attention == 'pre-rnn' or not use_teacher_forcing \
                or self.use_positional_attention:
            unrolling = True
        else:
            unrolling = False

        if unrolling:
            symbols = None
            for di in range(max_length):
                # We always start with the SOS symbol as input. We need to add extra dimension of length 1 for the number of decoder steps (1 in this case)
                # When we use teacher forcing, we always use the target input.
                if di == 0 or use_teacher_forcing:
                    decoder_input = inputs[:, di].unsqueeze(1)
                # If we don't use teacher forcing (and we are beyond the first
                # SOS step), we use the last output as new input
                else:
                    decoder_input = symbols

                # Perform one forward step
                decoder_output, decoder_hidden, step_attn, pos_attn = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs, function=function, unroll_step=di)
                # Remove the unnecessary dimension.
                step_output = decoder_output.squeeze(1)
                # Get the actual symbol
                symbols = decode(di, step_output, step_attn, pos_attn)

        else:
            # Remove last token of the longest output target in the batch. We don't have to run the last decoder step where the teacher forcing input is EOS (or the last output)
            # It still is run for shorter output targets in the batch
            decoder_input = inputs[:, :-1]

            # Forward step without unrolling
            decoder_output, decoder_hidden, attn, pos_attn = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn, pos_attn)

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = torch.tensor(lengths, device=device)

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h)
                                    for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden,
                       encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention or self.use_positional_attention:
            if encoder_outputs is None:
                raise ValueError(
                    "Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError(
                    "Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.tensor([self.sos_id] * batch_size, dtype=torch.long,
                                  device=device).view(batch_size, 1)

            max_length = self.max_length
        else:
            # minus the start of sequence symbol
            max_length = inputs.size(1) - 1

        return inputs, batch_size, max_length
