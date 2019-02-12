
import torchtext
import torch
import csv
from tqdm import tqdm


def combine_src_tgt_files(src_path, tgt_path, output_path):
    with open(src_path) as src_file, \
            open(tgt_path) as tgt_file, \
            open(output_path, 'w') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        for data, targets in tqdm(zip(src_file, tgt_file)):
            data = data.strip()
            targets = targets.strip()
            writer.writerow([data, targets])


if __name__ == "__main__":

    from machine.dataset import SourceField, TargetField

    # train
    combine_src_tgt_files('pcfg-attention/data/pcfg_set/100K/random_split/train.src',
                          'pcfg-attention/data/pcfg_set/100K/random_split/train.tgt',
                          'pcfg-attention/data/pcfg_set/100K/train.tsv')
    # dev
    combine_src_tgt_files('pcfg-attention/data/pcfg_set/100K/random_split/dev.src',
                          'pcfg-attention/data/pcfg_set/100K/random_split/dev.tgt',
                          'pcfg-attention/data/pcfg_set/100K/dev.tsv')
    # test
    combine_src_tgt_files('pcfg-attention/data/pcfg_set/100K/random_split/test.src',
                          'pcfg-attention/data/pcfg_set/100K/random_split/test.tgt',
                          'pcfg-attention/data/pcfg_set/100K/test.tsv')

    use_output_eos = False
    src = SourceField(batch_first=True)
    tgt = TargetField(include_eos=use_output_eos, batch_first=True)
    tabular_data_fields = [('src', src), ('tgt', tgt)]

    max_len = 50

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    train = torchtext.data.TabularDataset(
        path='pcfg-attention/data/pcfg_set/10K/train.tsv', format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter)
