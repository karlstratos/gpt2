import argparse
import datasets

from collections import Counter
from datasets import load_dataset
from file_handling import mkdir_optional


def write_file(dirname, filename, lines, show_top=False):
    path = dirname + '/' + filename + '.txt'
    vocab = Counter()
    with open(path, 'w') as f:
        for line in lines:
            vocab.update(Counter(line.split()))
            f.write(line)  # Already ends in '\n'
    string = '{:s}: {:d} lines, {:d} toks, {:d} unique toks'.format(
        filename, len(lines), sum(vocab.values()), len(vocab))
    if show_top:
        string += ' ({:s})'.format(
            ' '.join([tok for tok, _ in vocab.most_common(10)]))
        string += ', min freq {:d}'.format(min(vocab.values()))
    print(string)


def main(args):
    dirname = 'data/' + args.config
    mkdir_optional(dirname)
    dataset = load_dataset('wikitext', args.config)

    print('wikitext: ', args.config)
    write_file(dirname, 'train', dataset['train']['text'], show_top=True)
    write_file(dirname, 'val', dataset['validation']['text'])
    write_file(dirname, 'test', dataset['test']['text'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='wikitext-2-raw-v1',
                        choices=['wikitext-103-v1', 'wikitext-2-v1',
                                 'wikitext-103-raw-v1', 'wikitext-2-raw-v1'])
    args = parser.parse_args()

    main(args)
