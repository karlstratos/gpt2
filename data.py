import os
import pickle
import torch

from itertools import chain
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


def get_loaders(model_name, data_dir, max_length, seed, batch_size, no_shuffle,
                num_workers, is_distributed):
    dataset_train = LMDataset(data_dir + '/train.txt', model_name, max_length)
    dataset_val = LMDataset(data_dir + '/val.txt', model_name, max_length)
    dataset_test = LMDataset(data_dir + '/test.txt', model_name, max_length)

    sampler_train = DistributedSampler(dataset_train, shuffle=True, seed=seed) \
        if is_distributed else None
    loader_train = DataLoader(dataset_train, batch_size,
                              shuffle=(sampler_train is None and
                                       not no_shuffle),
                              sampler=sampler_train,
                              num_workers=num_workers)

    sampler_val = DistributedSampler(dataset_val, shuffle=False) \
        if is_distributed else None
    loader_val = DataLoader(dataset_val, batch_size, shuffle=False,
                            sampler=sampler_val, num_workers=num_workers)

    sampler_test = DistributedSampler(dataset_test, shuffle=False) \
        if is_distributed else None
    loader_test = DataLoader(dataset_test, batch_size, shuffle=False,
                             sampler=sampler_test, num_workers=num_workers)

    return loader_train, loader_val, loader_test


class LMDataset(Dataset):

    def __init__(self, path, model_name, length=1024):
        self.pickle_path = str(Path(path).with_suffix('')) + \
            '_{:s}_len{:d}.pickle'.format(model_name, length)
        try:
            with open(self.pickle_path, 'rb') as f:
                self.seq_pairs, self.vocab_size = pickle.load(f)
        except:
            self.seq_pairs, self.vocab_size = tokenize_file(path, model_name,
                                                            length)
            with open(self.pickle_path, 'wb') as f:
                pickle.dump((self.seq_pairs, self.vocab_size), f)

    def __len__(self):
        return len(self.seq_pairs)

    def __getitem__(self, index):
        input_ids, attention_mask = self.seq_pairs[index]
        return torch.tensor(input_ids), torch.BoolTensor(attention_mask), \
            torch.tensor(index)


def tokenize_file(path, model_name, length=1024):
    with open(path) as f:
        lines = f.readlines()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output = tokenizer(lines)
    output_cat = {key: list(chain(*output[key])) for key in output.keys()}
    total_length = len(output_cat['input_ids'])
    result = {key: [seq[t:t + length] for t in range(0, total_length, length)]
              for key, seq in output_cat.items()}

    # Pad the last sequence because we can.
    pad_size = length - len(result['attention_mask'][-1])
    pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None \
        else tokenizer.pad_token_id  # No padding used in pretrained GPT-2
    result['input_ids'][-1] += [pad_token_id] * pad_size
    result['attention_mask'][-1] += [0] * pad_size

    samples = [(result['input_ids'][i], result['attention_mask'][i])
               for i in range(len(result['input_ids']))]
    return samples, tokenizer.vocab_size
