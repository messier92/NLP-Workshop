# Debugging 1: Bert Model takes too long to train
# Download directly at https://github.com/huggingface/transformers/issues/136
# model = BertModel.from_pretrained('THE-PATH-OF-X')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pymagnitude import Magnitude
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from scipy import spatial
from sklearn.manifold import TSNE
from tensorboardcolab import TensorBoardColab
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm_notebook as tqdm

RED, BLUE = '#FF4136', '#0074D9'

def cosine_similarity(word1, word2):
    vector1, vector2 = glove_vectors.query(word1), glove_vectors.query(word2)
    return 1 - spatial.distance.cosine(vector1, vector2)

sentence = 'the quick brown fox jumps over the lazy dog'
words = sentence.split()

# We first turn this sentence into numbers by assigning each unique word an integer.
word2idx = {word: idx for idx, word in enumerate(sorted(set(words)))}
# Then, we turn each word in our sentence into its assigned index.
idxs = torch.LongTensor([word2idx[word] for word in sentence.split()])

# Next, we want to create an embedding layer.
# The embedding layer is a 2-D matrix of shape (n_vocab x embedding_dimension).
# If we apply our input list of indices to the embedding layer, each value in the input list of indices maps to that specific row of the embedding layer matrix.
# The output shape after applying the input list of indices to the embedding layer is another 2-D matrix of shape (n_words x embedding_dimension).

embedding_layer = nn.Embedding(num_embeddings=len(word2idx), embedding_dim=3)
embeddings = embedding_layer(idxs)
print(embeddings, embeddings.shape)

# https://nlp.stanford.edu/projects/glove/
# GloVe is an unsupervised learning algorithm for obtaining vector representations for words.
# Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.
# Download the .magnitude file from https://github.com/plasticityai/magnitude#installation

glove_vectors = Magnitude('glove6B/glove.6B.50d.magnitude')
glove_embeddings = glove_vectors.query(words)

word_pairs = [
    ('dog', 'cat'),
    ('tree', 'cat'),
    ('tree', 'leaf'),
    ('king', 'queen'),
    ('master', 'slave'),
    ('men', 'mars'),
    ('women', 'venus'),
    ('drain', 'swamp'),
    ('bird', 'stars')
]

for word1, word2 in word_pairs:
    print(f'Similarity between "{word1}" and "{word2}":\t{cosine_similarity(word1, word2):.2f}')

# Visualizing Embeddings
# We can demonstrate that embeddings carry semantic information by plotting them. However, because our embeddings are more than three dimensions, they are impossible to visualize.
# Therefore, we can use an algorithm called t-SNE to project the word embeddings to a lower dimension in order to plot them in 2-D.

ANIMALS = [
    'whale',
    'fish',
    'horse',
    'rabbit',
    'sheep',
    'lion',
    'dog',
    'cat',
    'tiger',
    'hamster',
    'pig',
    'goat',
    'lizard',
    'elephant',
    'giraffe',
    'hippo',
    'zebra',
]

HOUSEHOLD_OBJECTS = [
    'stapler',
    'screw',
    'nail',
    'tv',
    'dresser',
    'keyboard',
    'hairdryer',
    'couch',
    'sofa',
    'lamp',
    'chair',
    'desk',
    'pen',
    'pencil',
    'table',
    'sock',
    'floor',
    'wall',
]

tsne_words_embedded = TSNE(n_components=2).fit_transform(glove_vectors.query(ANIMALS + HOUSEHOLD_OBJECTS))

x, y = zip(*tsne_words_embedded)

fig, ax = plt.subplots(figsize=(10, 8))

for i, label in enumerate(ANIMALS + HOUSEHOLD_OBJECTS):
    if label in ANIMALS:
        color = BLUE
    elif label in HOUSEHOLD_OBJECTS:
        color = RED
        
    ax.scatter(x[i], y[i], c=color)
    ax.annotate(label, (x[i], y[i]))

ax.axis('off')

#plt.show()

# Context Embeddings
# GloVe and Fasttext are two examples of global embeddings, where the embeddings don't change even though the "sense" of the word might change given the context.
# This can be a problem for cases such as:
# A mouse stole some cheese.
# I bought a new mouse the other day for my computer.
# The word "mouse" can mean both an animal and a computer accessory depending on the context, yet for GloVe they would receive the same exact distributed representation. We can combat this by taking into account the surroudning words to create a context-sensitive embedding.
# Context embeddings such as Bert are really popular right now.

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('D:\\HackathonsAndStartups\\PyTorch Introduction\\Embddings_Example\\bert-base-uncased.tar.gz')
model.eval()

def to_bert_embeddings(text, return_tokens=False):
    if isinstance(text, list):
        # Already tokenized
        tokens = tokenizer.tokenize(' '.join(text))
    else:
        # Need to tokenize
        tokens = tokenizer.tokenize(text)
        
    tokens_with_tags = ['[CLS]'] + tokens + ['[SEP]']
    indices = tokenizer.convert_tokens_to_ids(tokens_with_tags)

    out = model(torch.LongTensor(indices).unsqueeze(0))
    
    # Concatenate the last four layers and use that as the embedding
    # source: https://jalammar.github.io/illustrated-bert/
    embeddings_matrix = torch.stack(out[0]).squeeze(1)[-4:]  # use last 4 layers
    embeddings = []
    for j in range(embeddings_matrix.shape[1]):
        embeddings.append(embeddings_matrix[:, j, :].flatten().detach().numpy())
        
    # Ignore [CLS] and [SEP]
    embeddings = embeddings[1:-1]
        
    if return_tokens:
        assert len(embeddings) == len(tokens)
        return embeddings, tokens
    
    return embeddings

words_sentences = [
    ('mouse', 'I saw a mouse run off with some cheese.'),
    ('mouse', 'I bought a new computer mouse yesterday.'),
    ('cat', 'My cat jumped on the bed.'),
    ('keyboard', 'My computer keyboard broke when I spilled juice on it.'),
    ('dessert', 'I had a banana fudge sunday for dessert.'),
    ('dinner', 'What did you eat for dinner?'),
    ('lunch', 'Yesterday I had a bacon lettuce tomato sandwich for lunch. It was tasty!'),
    ('computer', 'My computer broke after the motherdrive was overloaded.'),
    ('program', 'I like to program in Java and Python.'),
    ('pasta', 'I like to put tomatoes and cheese in my pasta.'),
]
words = [words_sentence[0] for words_sentence in words_sentences]
sentences = [words_sentence[1] for words_sentence in words_sentences]

embeddings_lst, tokens_lst = zip(*[to_bert_embeddings(sentence, return_tokens=True) for sentence in sentences])
words, tokens_lst, embeddings_lst = zip(*[(word, tokens, embeddings) for word, tokens, embeddings in zip(words, tokens_lst, embeddings_lst) if word in tokens])

# Convert tuples to lists
words, tokens_lst, tokens_lst = map(list, [words, tokens_lst, tokens_lst])

target_indices = [tokens.index(word) for word, tokens in zip(words, tokens_lst)]

target_embeddings = [embeddings[idx] for idx, embeddings in zip(target_indices, embeddings_lst)]

tsne_words_embedded = TSNE(n_components=2).fit_transform(target_embeddings)
x, y = zip(*tsne_words_embedded)

fig, ax = plt.subplots(figsize=(5, 10))

for word, tokens, x_i, y_i in zip(words, tokens_lst, x, y):
    ax.scatter(x_i, y_i, c=RED)
    ax.annotate(' '.join([f'$\\bf{x}$' if x == word else x for x in tokens]), (x_i, y_i))

ax.axis('off')
plt.show()
