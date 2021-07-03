import gzip
import csv
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# filename = 'datasets/names_test.csv.gz'
# with gzip.open(filename, 'rt') as f:
#     reader = csv.reader((f))
#     rows = list(reader)
#     print(rows)
#


class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'datasets/names_train.csv.gz' if is_train_set else 'datasets/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dic = self.getCountryDict()
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dic[self.countries[index]]

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dic = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dic[country_name] = idx

        return country_dic

    def idx2country(self, index):
        return self.country_list[index]

    def getCountriesNum(self):
        return self.country_num


HIDDEN_SINE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_EPOCH = 100
N_CHARS = 128
USE_GPU = False

train_set = NameDataset(is_train_set=True)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_set = NameDataset(is_train_set=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device('cuda；0')
        tensor = tensor.to(device)
    return tensor
    pass


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers,
                                bidriectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)

        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)

        gru_input = pack_padded_sequence(embedding, seq_lengths)

        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        fc_output = self.fc(hidden_cat)

        return fc_output


def make_tensors(names, countries):
    sequences_and_lengths = [names2list(name) for name in names]
    name_sequences = [sl[0] for sl in sequences_and_lengths]
