import os
import argparse
import logging

import torch
import torchtext
from collections import OrderedDict

from machine.trainer import SupervisedTrainer
from machine.models import EncoderRNN, DecoderRNN, Seq2seq
from machine.loss import NLLLoss
from machine.metrics import WordAccuracy, SequenceAccuracy, FinalTargetAccuracy
from machine.dataset import SourceField, TargetField
from machine.util.checkpoint import Checkpoint
from machine.dataset.get_standard_iter import get_standard_iter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
IGNORE_INDEX = -1
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

# Prepare logging and data set
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))


def prepare_iters():

    use_output_eos = False
    src = SourceField(batch_first=True)
    tgt = TargetField(batch_first=True, include_eos=use_output_eos)
    tabular_data_fields = [('src', src), ('tgt', tgt)]

    max_len = 50

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    # generate training and testing data
    train = get_standard_iter(torchtext.data.TabularDataset(
        path='pcfg-attention/.data/pcfg_set/10K/train.tsv', format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter), batch_size=128)

    dev = get_standard_iter(torchtext.data.TabularDataset(
        path='pcfg-attention/.data/pcfg_set/10K/dev.tsv', format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter), batch_size=128)

    monitor_data = OrderedDict()
    m = get_standard_iter(torchtext.data.TabularDataset(
        path='pcfg-attention/.data/pcfg_set/10K/test.tsv', format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter), batch_size=128)
    monitor_data['test'] = m

    return src, tgt, train, dev, monitor_data


def initialize_model(src, tgt, train):
    # build vocabulary
    src.build_vocab(train.dataset, max_size=1000)
    tgt.build_vocab(train.dataset, max_size=1000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Initialize model
    hidden_size = 50
    embedding_size = 50
    max_len = 100
    decoder_hidden_size = hidden_size
    encoder = EncoderRNN(len(src.vocab), max_len, hidden_size, embedding_size,
                         n_layers=1,
                         rnn_cell='lstm',
                         variable_lengths=True)
    decoder = DecoderRNN(len(tgt.vocab), max_len, decoder_hidden_size,
                         n_layers=1,
                         attention_method='mlp',
                         full_focus=True,
                         rnn_cell='lstm',
                         eos_id=tgt.eos_id,
                         sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)
    seq2seq.to(device)

    return seq2seq, input_vocab, output_vocab


def prepare_losses_and_metrics(pad, eos):
    # Prepare loss and metrics
    losses = [NLLLoss(ignore_index=pad)]
    loss_weights = [1.]

    for loss in losses:
        loss.to(device)

    metrics = []
    metrics.append(WordAccuracy(ignore_index=pad))
    metrics.append(SequenceAccuracy(ignore_index=pad))
    return losses, loss_weights, metrics


if __name__ == "__main__":
    src, tgt, train, dev, monitor_data = prepare_iters()

    # Prepare model
    seq2seq, input_vocab, output_vocab = initialize_model(src, tgt, train)

    pad = output_vocab.stoi[tgt.pad_token]
    eos = tgt.eos_id
    sos = tgt.SYM_EOS
    unk = tgt.unk_token

    # Prepare training
    losses, loss_weights, metrics = prepare_losses_and_metrics(pad, eos)
    trainer = SupervisedTrainer(expt_dir='pcfg-attention/runs')

    # Train
    seq2seq, _ = trainer.train(seq2seq, train,
                               num_epochs=5, dev_data=dev, monitor_data=monitor_data,
                               losses=losses, metrics=metrics, loss_weights=loss_weights)
