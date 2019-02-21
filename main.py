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

from utils import generate_filename_from_options, TensorboardCallback, EarlyStoppingCallback, ReduceLRonPlateauCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
IGNORE_INDEX = -1
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

# Prepare logging and data set
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))


def init_argparser():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs (default, 10)', default=10)
    parser.add_argument('--optim', type=str, help='Choose optimizer',
                        choices=['adam', 'adadelta', 'adagrad',
                                 'adamax', 'rmsprop', 'sgd'],
                        default='sgd')
    parser.add_argument('--max_len', type=int,
                        help='Maximum sequence length', default=100)
    parser.add_argument(
        '--rnn_cell', help="Chose type of rnn cell", default='lstm')
    parser.add_argument('--bidirectional', action='store_true',
                        help="Flag for bidirectional encoder")
    parser.add_argument('--embedding_size', type=int,
                        help='Embedding size', default=128)
    parser.add_argument('--hidden_size', type=int,
                        help='Hidden layer size', default=128)
    parser.add_argument('--n_layers', type=int,
                        help='Number of RNN layers in both encoder and decoder', default=1)
    parser.add_argument('--src_vocab', type=int,
                        help='source vocabulary size', default=600)
    parser.add_argument('--tgt_vocab', type=int,
                        help='target vocabulary size', default=600)
    parser.add_argument('--dropout_p_encoder', type=float,
                        help='Dropout probability for the encoder', default=0.2)
    parser.add_argument('--dropout_p_decoder', type=float,
                        help='Dropout probability for the decoder', default=0.2)
    parser.add_argument('--teacher_forcing_ratio', type=float,
                        help='Teacher forcing ratio', default=0.0)

    # Attention arguments
    parser.add_argument(
        '--attention', choices=['pre-rnn', 'post-rnn'], default=False)
    parser.add_argument('--attention_method',
                        choices=['dot', 'mlp', 'concat'], default=None)
    parser.add_argument('--positional_attention_method',
                        choices=['dot', 'mlp', 'concat'], default=None)
    parser.add_argument('--positional_attention',
                        choices=['dot', 'mlp', 'concat'], default=None)

    parser.add_argument('--full_focus', action='store_true')

    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=32)
    parser.add_argument('--eval_batch_size', type=int,
                        help='Batch size', default=128)
    parser.add_argument(
        '--lr', type=float, help='Learning rate, recommended settings.\nrecommended settings: adam=0.001 adadelta=1.0 adamax=0.002 rmsprop=0.01 sgd=0.1', default=0.01)
    parser.add_argument('--use_output_eos', action='store_true',
                        help='Use end of sequence token during training and evaluation')

    # Data management
    parser.add_argument('--load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--save_every', type=int,
                        help='Every how many batches the model should be saved', default=100)
    parser.add_argument('--print_every', type=int,
                        help='Every how many batches to print results', default=100)
    parser.add_argument('--resume-training', action='store_true',
                        help='Indicates if training has to be resumed from the latest checkpoint')
    parser.add_argument('--log-level', default='info', help='Logging level.')
    parser.add_argument('--mini', action='store_true',
                        help="Flag for using mini dataset")
    parser.add_argument('--write_logs', action='store_true',
                        help="Flag for writing logs after training")
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')
    return parser


def validate_options(parser, opt):
    if opt.resume_training and not opt.load_checkpoint:
        parser.error(
            'load_checkpoint argument is required to resume training from checkpoint')

    if not opt.attention and opt.attention_method:
        parser.error(
            "Attention method provided, but attention is not turned on")

    if opt.attention and not opt.attention_method:
        logging.info("No Attention method provided. Using DOT method.")
        opt.attention_method = 'dot'

    return opt


def prepare_iters(opt):

    src = SourceField(batch_first=True)
    tgt = TargetField(batch_first=True, include_eos=opt.use_output_eos)
    tabular_data_fields = [('src', src), ('tgt', tgt)]

    max_len = opt.max_len

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    ds = '100K'
    if opt.mini:
        ds = '10K'

    # generate training and testing data
    train = get_standard_iter(torchtext.data.TabularDataset(
        path='data/pcfg_set/{}/train.tsv'.format(ds), format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter), batch_size=opt.batch_size)

    dev = get_standard_iter(torchtext.data.TabularDataset(
        path='data/pcfg_set/{}/dev.tsv'.format(ds), format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter), batch_size=opt.eval_batch_size)

    monitor_data = OrderedDict()
    m = get_standard_iter(torchtext.data.TabularDataset(
        path='data/pcfg_set/{}/test.tsv'.format(ds), format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter), batch_size=opt.eval_batch_size)
    monitor_data['Test'] = m

    return src, tgt, train, dev, monitor_data


def initialize_model(opt, src, tgt, train):
    # build vocabulary
    src.build_vocab(train.dataset, max_size=opt.src_vocab)
    tgt.build_vocab(train.dataset, max_size=opt.tgt_vocab)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Initialize model
    hidden_size = opt.hidden_size
    decoder_hidden_size = hidden_size * 2 if opt.bidirectional else hidden_size
    encoder = EncoderRNN(len(src.vocab), opt.max_len, hidden_size, opt.embedding_size,
                         dropout_p=opt.dropout_p_encoder,
                         n_layers=opt.n_layers,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         variable_lengths=True)
    decoder = DecoderRNN(len(tgt.vocab), opt.max_len, decoder_hidden_size,
                         dropout_p=opt.dropout_p_decoder,
                         n_layers=opt.n_layers,
                         use_attention=opt.attention,
                         attention_method=opt.attention_method,
                         full_focus=opt.full_focus,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id)

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


def train_pcfg_model():

    # Create command line argument parser and validate chosen options
    parser = init_argparser()
    opt = parser.parse_args()
    opt = validate_options(parser, opt)
    opt.file_name = generate_filename_from_options(opt)

    # Seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # Prepare data
    src, tgt, train, dev, monitor_data = prepare_iters(opt)

    # Prepare model
    seq2seq, input_vocab, output_vocab = initialize_model(opt, src, tgt, train)

    pad = output_vocab.stoi[tgt.pad_token]
    eos = tgt.eos_id
    sos = tgt.SYM_EOS
    unk = tgt.unk_token

    # Prepare training
    losses, loss_weights, metrics = prepare_losses_and_metrics(pad, eos)
    run_folder = 'runs/' + opt.file_name
    model_folder = 'models/'+opt.file_name
    trainer = SupervisedTrainer(expt_dir=model_folder)
    checkpoint_path = os.path.join(model_folder, opt.load_checkpoint
                                   ) if opt.resume_training else None

    # custom callbacks to log to tensorboard and do early stopping
    custom_cbs = [TensorboardCallback(
        run_folder), EarlyStoppingCallback(patience=20, monitor='eval_metrics',
                                           lm_name='Sequence Accuracy', minimize=False),
        ReduceLRonPlateauCallback(factor=0.5)]

    # Train
    seq2seq, logs = trainer.train(seq2seq, train,
                                  num_epochs=opt.epochs, dev_data=dev,
                                  monitor_data=monitor_data, optimizer=opt.optim,
                                  teacher_forcing_ratio=opt.teacher_forcing_ratio,
                                  learning_rate=opt.lr,
                                  resume_training=opt.resume_training,
                                  checkpoint_path=checkpoint_path,
                                  losses=losses, metrics=metrics, loss_weights=loss_weights,
                                  checkpoint_every=opt.save_every, print_every=opt.print_every,
                                  custom_callbacks=custom_cbs)

    if opt.write_logs:
        logs.write_to_file(run_folder+'/logs')


if __name__ == "__main__":
    train_pcfg_model()
