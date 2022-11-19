import json
from utils import *
from torch.utils.data import DataLoader, Dataset


# # class IndexedDataset(Dataset):
# #     def __init__(self, x, y):
# #         self.x = x
# #         self.y = y

# #     def __getitem__(self, index):
# #         return self.x[index], self.y[index]

# #     def __len__(self):
# #         return len(self.x)

# # def get_dataloader(dataset, batch_size, shuffle):
# #     return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
# class IndexedDataset(Dataset):
#     def __init__(self, x, y):
#         super().__init__()
#         self.x = x
#         self.y = y
#         self.xlen = [ len(_x_) for _x_ in x ]
#         self.ylen = [ len(_y_) for _y_ in y ]

#     def __getitem__(self, index):
#         return self.x[index], self.y[index], self.xlen[index], self.ylen[index]
    
#     def __len__(self):
#         return len(self.x)

# def pad_collate(batch):
#     (xx, yy, x_lens, y_lens) = zip(*batch)

#     x_tensor = pad_sequence(xx, batch_first=True, padding_value=0)
#     y_tensor = pad_sequence(yy, batch_first=True, padding_value=0)

#     return x_tensor, x_lens, y_tensor, y_lens

# def read_episodes(file):
#     with open(file, 'r') as f:
#         data = json.load(f)
#     return data['train'],data['valid_seen']


# def flatten_episodes(data):
#     episode_list =[]
#     output_seq =[]
#     for i in range(len(data)):
#         seq_text = []
#         action_target = []
#         for j in data[i]:
#             seq_text.append(j[0])
#             action_target.append(j[1])
#         episode_list.append(seq_text)
#         output_seq.append(action_target)
#     return episode_list, output_seq

# def concat_list(flat_data):
#     result = []
#     for i in flat_data:
#         i = [' '.join(i)]
#         result.append(i)
#     return result

# def encode_data(flat_episode, flat_output, v2i, t2i, a2i):
#     n_episodes = len(flat_episode)
#     y = [[] for _ in range(n_episodes)]
#     x=[[] for _ in range(n_episodes)]
#     idx=0
#     for episode,output_seq in zip(flat_episode,flat_output):
#         instruction = episode
#         classes = output_seq
#         instruction[0] = preprocess_string(instruction[0])
#         for word in instruction[0].split():
#             if len(word) > 0:
#                 x[idx].append(v2i[word] if word in v2i else v2i["<unk>"])
#         x[idx].insert(0,v2i["<start>"])
#         x[idx].append(v2i["<end>"]) 
#         y[idx]=[(a2i[word[0]],t2i[word[1]]) for word in classes]
#         y[idx].insert(0,(a2i["<start>"],t2i["<start>"])) 
#         y[idx].append((a2i["<end>"],t2i["<end>"]))
#         idx += 1
#     return x,y

# # def pad_collate(batch):
# #   (xx, yy) = zip(*batch)
# #   x_lens = [len(x) for x in xx]
# #   y_lens = [len(y) for y in yy]

# #   xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
# #   yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

# #   return xx_pad, yy_pad, x_lens, y_lens

train_data, val_data =  read_episodes('lang_to_sem_data.json')
vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_data, vocab_size = 3000)
actions_to_index, index_to_actions, targets_to_index, index_to_targets, max_len = build_output_tables(train_data)
train_x, train_y = flatten_episodes(train_data)
val_x, val_y = flatten_episodes(val_data)
train_set = concat_list(train_x)
val_set = concat_list(val_x)
x_train, y_train = encode_data(train_set, train_y, vocab_to_index, targets_to_index, actions_to_index)
x_val, y_val= encode_data(val_set, val_y, vocab_to_index, targets_to_index, actions_to_index)

train_dataset = CustomDataset([torch.from_numpy(np.array(xi)) for xi in x_train],[torch.from_numpy(np.array(yi)) for yi in y_train])
train_loader = DataLoader(dataset = train_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate)

val_dataset = CustomDataset([torch.from_numpy(np.array(xi)) for xi in x_val],[torch.from_numpy(np.array(yi)) for yi in y_val])
val_loader = DataLoader(dataset = val_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate)

print(max_len)
# for (a,b,c,d) in val_loader:
#     print(a.size())
#     print(len(b))
#     print(c.size())
#     print(len(d))
#     break








