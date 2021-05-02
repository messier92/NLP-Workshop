# Debugging : RuntimeError: Expected object of scalar type Long but got scalar type Int for argument #2 'target' in call to _thnn_nll_loss_forward
# Debugging : RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.cuda.IntTensor instead (while checking arguments for embedding)
# Debugging : Just add ".long()" to the respective argument indices instead
import string
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
from boltons.iterutils import windowed
from tqdm import tqdm, tqdm_notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'articles/articles.jsonl'

def load_data(path, sequence_length=125):
    texts = pd.read_json(path).text.sample(100).str.lower().tolist()
    chars_windowed = [list(windowed(text, sequence_length)) for text in texts]
    all_chars_windowed = [sublst for lst in chars_windowed for sublst in lst]
    filtered_good_chars = [
        sequence for sequence in tqdm_notebook(all_chars_windowed) 
        if all(char in string.printable for char in sequence)
    ]
    return filtered_good_chars


def get_unique_chars(sequences):
    return {sublst for lst in sequences for sublst in lst}


def create_char2idx(sequences):
    unique_chars = get_unique_chars(sequences)
    return {char: idx for idx, char in enumerate(sorted(unique_chars))}


def encode_sequence(sequence, char2idx):
    return [char2idx[char] for char in sequence]


def encode_sequences(sequences, char2idx):
    return np.array([
        encode_sequence(sequence, char2idx) 
        for sequence in tqdm_notebook(sequences)
    ])

# Text pre-processing
class Sequences(Dataset):
    def __init__(self, path, sequence_length=125):
        self.sequences = load_data(DATA_PATH, sequence_length=sequence_length)
        self.vocab_size = len(get_unique_chars(self.sequences))
        self.char2idx = create_char2idx(self.sequences)
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.encoded = encode_sequences(self.sequences, self.char2idx)
        
    def __getitem__(self, i):
        return self.encoded[i, :-1], self.encoded[i, 1:]
    
    def __len__(self):
        return len(self.encoded)

class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dimension=100,
        hidden_size=128, 
        n_layers=1,
        device='cpu',
    ):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device
        
        self.encoder = nn.Embedding(vocab_size, embedding_dimension)
        self.rnn = nn.GRU(
            embedding_dimension,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
    def init_hidden(self, batch_size):
        return torch.randn(self.n_layers, batch_size, self.hidden_size).to(self.device)

    # Give it one character and calculate the loss per character
    def forward(self, input_, hidden):
        encoded = self.encoder(input_)
        output, hidden = self.rnn(encoded.unsqueeze(1), hidden)
        output = self.decoder(output.squeeze(1))
        return output, hidden
    
dataset = Sequences(DATA_PATH, sequence_length=128)
len(dataset)
train_loader = DataLoader(dataset, batch_size=4096)

model = RNN(vocab_size=dataset.vocab_size, device=device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001,
)

print(model)
print()
print('Trainable parameters:')
print('\n'.join([' * ' + x[0] for x in model.named_parameters() if x[1].requires_grad]))

# The training loop
model.train()
train_losses = []
for epoch in range(50):
    progress_bar = tqdm_notebook(train_loader, leave=False)
    losses = []
    total = 0

    for inputs, targets in progress_bar:
        batch_size = inputs.size(0)
        hidden = model.init_hidden(batch_size)

        model.zero_grad()
        
        loss = 0
        # for every character in our training loop, apply the model to that ONE character
        # apply the criterion on that character
        for char_idx in range(inputs.size(1)):
            output, hidden = model((inputs[:, char_idx].to(device)).long(), hidden)
            loss += criterion(output, (targets[:, char_idx].to(device)).long())

        loss.backward()

        optimizer.step()
        
        avg_loss = loss.item() / inputs.size(1)
        
        progress_bar.set_description(f'Loss: {avg_loss:.3f}')
        
        losses.append(avg_loss)
        total += 1
    
    epoch_loss = sum(losses) / total
    train_losses.append(epoch_loss)
        
    tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')


def pretty_print(text):
    """Wrap text for nice printing."""
    to_print = ''
    for paragraph in text.split('\n'):
        to_print += '\n'.join(wrap(paragraph))
        to_print += '\n'
    print(to_print)

temperature = 1.0

model.eval()
seed = '\n'

# Generate a new text from the input article 
text = ''
with torch.no_grad():
    batch_size = 1
    hidden = model.init_hidden(batch_size)
    last_char = dataset.char2idx[seed]
    for _ in range(1000):
        output, hidden = model(torch.LongTensor([last_char]).to(device), hidden)
        
        distribution = output.squeeze().div(temperature).exp()
        guess = torch.multinomial(distribution, 1).item()
        
        last_char = guess
        text += dataset.idx2char[guess]
        
pretty_print(text)
