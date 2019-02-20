def generate_filename_from_options(opt):
    """
    Pass in options obtained from argparse and generates a filename for the run and models
    """
    fs = '{}_emb_{}_hid_{}_de_{}_dd_{}_n_lyrs_{}_lr_{}_maxlen_{}_teach_{}'.format(
        opt.rnn_cell,
        opt.embedding_size, opt.hidden_size,
        opt.dropout_p_encoder, opt.dropout_p_decoder,
        opt.n_layers, opt.lr, opt.max_len, opt.teacher_forcing_ratio)

    if opt.optim is not None:
        fs += '_{}'.format(opt.optim)
    if opt.use_output_eos:
        fs += '_use_eos'
    if opt.attention:
        if opt.attention == 'pre-rnn':
            fs += '_pre'
        if opt.attention == 'post-rnn':
            fs += '_post'
        fs += '_att_{}'.format(opt.attention_method)
        if opt.full_focus:
            fs += '_ff'
    if opt.bidirectional:
        fs += '_bidirect'
    if opt.mini:
        fs += '_mini'

    return fs
