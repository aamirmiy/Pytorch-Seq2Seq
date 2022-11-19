import re
import torch
import numpy as np
from collections import Counter
import json
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device

def read_episodes(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data['train'],data['valid_seen']

def preprocess_string(s):
    s=s.lower()
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    inst_count = 0
    for episode in train:
        for inst, _ in episode:
            inst = preprocess_string(inst)
            padded_len = 2  # start/end
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
            padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
        #int(np.max(padded_lens))
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    max_length=0
    for episode in train:
        for _, outseq in episode:
            max_length = max(max_length, len(outseq)+2)
            a, t = outseq
            actions.add(a)
            targets.add(t)
            
    actions_to_index = {a: i+3 for i, a in enumerate(actions)}
    targets_to_index = {t: i+3 for i, t in enumerate(targets)}
    actions_to_index["<pad>"] = 0
    actions_to_index["<start>"] = 1
    actions_to_index["<end>"] = 2

    targets_to_index["<pad>"] = 0
    targets_to_index["<start>"] = 1
    targets_to_index["<end>"] = 2
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets, max_length

def prefix_match(predicted_labels, gt_labels):
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1 

    seq_length = len(gt_labels)
    
    for i in range(seq_length):
        if torch.any(predicted_labels[i] != gt_labels[i]):
            break
    
    pm = (1.0 / seq_length) * i

    return pm


def flatten_episodes(data):
    episode_list =[]
    output_seq =[]
    for i in range(len(data)):
        seq_text = []
        action_target = []
        for j in data[i]:
            seq_text.append(j[0])
            action_target.append(j[1])
        episode_list.append(seq_text)
        output_seq.append(action_target)
    return episode_list, output_seq

def concat_list(flat_data):
    result = []
    for i in flat_data:
        i = [' '.join(i)]
        result.append(i)
    return result

def encode_data(flat_episode, flat_output, v2i, t2i, a2i):
    n_episodes = len(flat_episode)
    y = [[] for _ in range(n_episodes)]
    x=[[] for _ in range(n_episodes)]
    idx=0
    for episode,output_seq in zip(flat_episode,flat_output):
        instruction = episode
        classes = output_seq
        instruction[0] = preprocess_string(instruction[0])
        for word in instruction[0].split():
            if len(word) > 0:
                x[idx].append(v2i[word] if word in v2i else v2i["<unk>"])
        x[idx].insert(0,v2i["<start>"])
        x[idx].append(v2i["<end>"])
        
        y[idx]=[(a2i[word[0]],t2i[word[1]]) for word in classes]
        y[idx].insert(0,(a2i["<start>"],t2i["<start>"])) 
        y[idx].append((a2i["<end>"],t2i["<end>"]))
        idx += 1
    return x,y

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

  return xx_pad, x_lens, yy_pad, y_lens