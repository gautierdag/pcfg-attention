def generate_filename_from_options(opt):
    """
    Pass in options obtained from argparse and generates a filename for the run and models
    """
    fs = '{}_emb_{}_hid_{}_de_{}_dd_{}_n_lyrs_{}_lr_{}'.format(
        opt.rnn_cell,
        opt.embedding_size, opt.hidden_size,
        opt.dropout_p_encoder, opt.dropout_p_decoder,
        opt.n_layers, opt.lr)

    if opt.optim is not None:
        fs += '_{}'.format(opt.optim)
    if opt.attention:
        fs += '_att_{}'.format(opt.attention_method)
    if opt.positional_attention:
        fs += '_pos_att_{}'.format(opt.positioning_generator_size)
    if opt.attention and opt.positional_attention:
        fs += '_mix_{}'.format(opt.attention_mixer)
    if opt.bidirectional:
        fs += '_bidirect'
    if opt.mini:
        fs += '_mini'

    return fs
