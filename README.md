# NLP-Workshop
https://www.youtube.com/watch?v=4jROlXH9Nvc A Deep Dive into NLP with PyTorch

A Deep Dive into NLP with PyTorch
https://www.youtube.com/watch?v=4jROlXH9Nvc
Table of Contents
1. Introducing PyTorch
1a. Static vs. dynamic computation graphs
1b. PyTorch basic operations
2. Training Networks
2a. Training flow
2b. Loss functions
2c. Gradient Descent
2d. Detailed PyTorch patterns
3. NLP Basics
3a. Examples of DL-NLP
3b. Text preprocessing
3c. Text representation
4. Embeddings
4a. Bag-of-words example
4b. word2vec,GloVe
4c. GloVe visualization
5. RNNs
5a. How RNNs work
5b. Vanishing and Exploding Gradient Problem
5c. Type of RNNs
5d. Char-RNN text generation example
5e. Sentiment classification example
6. Sequence Models
6a. Seq2Seq
6b. Attention
6c. Transformer
Goals of a deep learning library
1) Define a model, loss function, and learning rate optimizer (define the computational graph)
2) Support automatic differentiation







1a. Static vs. dynamic computation graphs
Static
•	“define-then-run”
•	Define the computational graph and then feed data to it
•	Easier to distribute over multiple machines
•	Ability to ship models independent of code
•	More complex to code and debug	Dynamic
•	“define-by-run”
•	Computational graph is defined on-the-fly
•	Ability to use a debugger to view data flow and check matrix shapes
•	Can handle variable sequence lengths
•	Easier to define some kinds of complex networks

1b. PyTorch basic operations
In [1]: import torch

In [2]: torch.tensor([1,2,3,]) # Create a tensor
Out [2]: tensor([1,2,3])

In [3]: _ * 2 # Broadcasting
Out [3]: tensor([2, 4, 6])

In [4]: _.to(‘cuda’) # Send the tensor to the GPU
Out[4]: tensor([2, 4, 6])

A graph is generated on the fly! 
x = torch.randn(1, 10)
prev_h = torch.randn(1, 20)

W_h = torch.randn(20, 20)
W_x = torch.randn(20, 10)
W_h.requires_grad = True
W_x.requires_grad = True

i2h = torch.mm(W_x, x.t())
h2h = torch.mm(W_h, prev_h.t())

next_h = i2h + h2h
next_h = next_h.tanh()

next_h.backward(torch.ones(20, 1))	 
  





2a. Training flow
 
 
 
2b. Common loss functions
Loss tells you how wrong your model is – and how your weights should be updated
 
2b. Using a Loss Function in PyTorch
criterion = nn.CrossEntropyLoss()

# Calculate how wrong the model is
loss = criterion(prediction, target)


2c. Gradient descent
Find a position in the graph to minimize the loss.
 
2d. PyTorch Dataset Class
class MyDataset(Dataset):
   def __init__(self):
          # Read your data file
          # Tokenize and clean text
          # Convert tokens to indices
   # must implement
   def __getitem__(self, i):
          return self.sequences[i], self.targets[i]
   # must implement
   def __len__(self):
          return len(self.sequences)

dataset = MyDataset()








2d. PyTorch Model Definition
class MyClassifier(nn.Module):
   def __init__(self):
          super(MyClassifier, self).__init__()
          self.fc1 = nn.Linear(128,32)
          self.fc2 = nn.Linear(32, 16)
          self.fc3 = nn.Linear(16, 1)

   def forward(self, inputs):
          x = F.relu(self.fc1(inputs))
          x = F.relu(self.fc2(x))
          x = F.sigmoid(self.fc3(x))
          return x

model = MyClassifier().to(‘cuda’)

2d. PyTorch Training Loop
for epoch in range(n_epochs):
   for inputs, target in loader:
       # Clean old gradients
       optimizer.zero_grad()
       
       # Forwards pass
       output = model(inputs)

       # Calculate how wrong the model is
       loss = criterion(output, target)

       # Perform gradient descent, backwards pass
       loss.backward()

       # Take a step in the right direction
       optimizer.step()











3a. Examples of DL-NLP
•	Language comprehension, tagging – Asking Alexa to order something from Amazon
•	Machine translation – Using Google Translate to translate from German to English
•	Text generation, summarization – Google News showing a summary of a news article
•	Named-entity recognition (NER) – Identifying products on a manufacturer’s website
•	Sentiment Analysis – Amazon detecting the sentiment of product reviews
3b. Text Preprocessing for NLP
In [1]: tokenize(‘the sneaky fox jumped over the dog’)
Out[1]: [‘the’, ‘sneaky’, ‘fox’, ‘jumped’, ‘over’, ‘the’, ‘dog’]

# Stopwords are the most common words in any natural language – they do
# not add much value to the meaning of the document
# such as “the”, “is”, “in”, “for”, “where”, “when”, “to”, “at”, etc...
In[2]: remove_stop_words(_)
Out[2]: [‘sneaky’, ‘fox’, ‘jumped’, ‘over’, ‘dog’]

In[3]: lemmatize(_)
Out[3]: [‘sneaky’, ‘fox’, ‘jump’, ‘over’, ‘dog’]

In[4]: replace_rare_words(_) # e.g. WordPiece
Out[4]: [‘###y’, ‘fox’, ‘jump’, ‘over’, ‘dog’]

3c. One-hot encoding 
Most classical method after pre-processing.
 
3c. Bag-of-words representation
1. Get the count of each word in the entire corpus
2. Compare the count of each word in a sentence compared to the entire corpus
  
 
4a. Bag-of-Words (35:23)
https://github.com/scoutbee/pytorch-nlp-notebooks/blob/develop/1_BoW_text_classification.ipynb
General Steps:
1. Load the .csv file

2. Sequence the dataset 
2a. Use CountVectorizer to get the frequency of each word (remove stop/common words)
2b. self.sequences refers to the INPUT (i.e. the reviews) use to train the model (X)
2c.  self.labels refer to the OUTPUT (i.e. the sentiments)
2d. perform tokenization
3. Get the processed dataset and use DataLoader on it, indicating the batch size. Save the results in train_loader
 
4. Define the Bag Of Words model classifier
5. Set the criterion and optimizer (Dynamically update the weights)
6. Train the model!  
7. Test the output!
4b. word2vec,GloVe, Embedding
 
Can enable PyTorch to learn about embeddings while it is training.

 
 









5a. RNNs
 
  

5b. Vanishing and Exploding Gradient Problem
 
One-to-One – no RNN, classic neural network
One-to-Many – One image input, Many captions output
Many-to-One – Many words (in a sentiment), one output (negative, neutral, positive)
Many-to-Many – Translate many English words (in a sentence) to another language i.e. German (in a sentence)
 
 
5c. Type of RNNs
 
 
 
•	Circle is Hadamard product (element-wise product)
•	sigma_c and sigma_h are both tanh

•	Types of gates
o	Forget gate: eg. when you are talking about Alice’s hobbies, but then you start talking about Bob, the forget gate can force the network to “forget” Alice’s hobbies to focus on Bob’s
o	Update gate: create candidate values that could be added to the next cell state and decide which ones to add
o	Output gate: to create next hidden state


•	Types of activation functions used
o	tanh: to normalize values between -1 and 1 for stability
o	sigmoid: to keep or remove parts (0 to completely remove, 1 to completely keep)

•	Why do we need both a cell state and a hidden state if they are both calculated recursively based on previous output?
o	Cell state is only additive or subtractive, so it is much better at keeping memories long-term than the hidden state
o	That said, it turns out to not be completely necessary to maintain both… (on to the GRU!)

 
 
1. Define __init__ and forward()
2. Make __init__ a superclass
3. Implement Embedding layer as encoder
4. Define RNN to be GRU
5. Define decoder that takes the output of the GRU shape and project it to the shape of our target
i.e. if we are doing classification we want 1 (1 or 0), if we want to distinguish between x number of classes then we put x
6. Forward – how we transform the input to the output- encode, update using rnn, decode
5d. Char-RNN text generation example
 
Character generation – predict the next letter from the previous one (feeds back at every character generated)
Apply GRU at every timestep and it will pass the hidden to itself


5e. Sentiment classification example
 
6. Sequence Models
6a. Seq2Seq
 
Feed the input text into an Encoder (GRU or simple RNN) and at each step, it constructs the next hidden state for each word and turn it into a context vector, and passes it into a decoder and generates an output translation. Both Encoder and Decoder can be RNNs.
 
 
 
 
s 
 
 
 
 

 
6b. Attention
Instead of feeding only the last hidden state, we take all the hidden state from the source and feed it into the decoder.
For the decoder, it takes the context at each timestamp and decide which is the most important at that frame to make the next decision.

 
Take the target and compare it to the previous hidden state. Give it a start and end token. Also give the model a maximum length so it doesn’t go on forever.
The encoder is always the same but the decoder gives it a different emphasis.
 
 
 












6c. Transformer
 
 
 
 
 
