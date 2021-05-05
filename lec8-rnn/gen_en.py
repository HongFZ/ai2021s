import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import itertools
import collections
import matplotlib.pyplot as plt

# Read in data
# df = pd.read_csv("Chinese_Names_Corpus_Gender（120W）.txt", header=2)
df = pd.read_csv("English_Cn_Name_Corpus（48W）.txt", header=None, names=["name"], skiprows=2)
names = df["name"].values

# Compute character frequency
chars = [list(name) for name in names]
chars_flatten = list(itertools.chain(*chars))
freq = collections.Counter(chars_flatten)
freq = pd.DataFrame(freq.items(), columns=["char", "freq"])
freq = freq.sort_values(by="freq", ascending=False)

# Power law (?)
char_rank = np.arange(freq.shape[0])
char_freq = freq["freq"].values
plt.plot(char_rank, char_freq)
plt.plot(np.log(1.0 + char_rank), np.log(char_freq))

# Prepare data
dict_size = 50
charset_size = dict_size + 1  # for EOS
dict = list(freq["char"].values[:dict_size])
dict_set = set(dict)
dat = list(filter(lambda name: set(name).issubset(dict_set), names))

# One-hot encoding
def char2index(char):
    return dict.index(char)

def name2index(name):
    return [char2index(char) for char in name]

def char2tensor(char):
    tensor = torch.zeros(1, charset_size)
    tensor[0, char2index(char)] = 1
    return tensor

def name2tensor(name):
    tensor = torch.zeros(len(name), 1, charset_size)
    for i, char in enumerate(name):
        tensor[i, 0, char2index(char)] = 1
    return tensor

def names2tensor(names):
    n = len(names)
    lens = [len(name) for name in names]
    max_len = np.max(lens)
    tensor = torch.zeros(max_len, n, charset_size)
    target = torch.zeros(max_len, n, dtype=int) + charset_size - 1
    for i in range(n):
        name = names[i]             # the i-th name
        for j in range(len(name)):  # the j-th character in the name
            tensor[j, i, char2index(name[j])] = 1
            if j < len(name) - 1:
                target[j, i] = char2index(name[j + 1])
    return tensor, np.array(lens), target

char2index("斯")
name2index("斯基")
char2tensor("斯")
name2tensor("斯基")
names2tensor(["斯基", "斯诺夫"])



# Build model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, input_size)
        self.o2o = nn.Linear(hidden_size + input_size, input_size)
        self.dropout = nn.Dropout(0.1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = torch.relu(self.i2h(input_combined))
        output = torch.relu(self.i2o(input_combined))
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.logsoftmax(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

n_hidden = 64
rnn = RNN(charset_size, n_hidden)
input = name2tensor("斯基")
hidden = rnn.init_hidden(batch_size=1)
output, next_hidden = rnn(input[0], hidden)



np.random.seed(123)
torch.random.manual_seed(123)
device = torch.device("cuda")
# device = torch.device("cpu")  # If no GPU on the machine

# train_id = np.random.choice(len(dat), 10000)
# train = [dat[i] for i in train_id]
train = dat

n = len(train)
n_hidden = 256
nepoch = 100
bs = 256

rnn = RNN(charset_size, n_hidden)
rnn = rnn.to(device=device)
opt = torch.optim.Adam(rnn.parameters(), lr=0.001)
train_ind = np.arange(n)
lossfn = nn.NLLLoss(reduction="none")
losses = []

t1 = time.time()
for k in range(nepoch):
    np.random.shuffle(train_ind)
    # Update on mini-batches
    for j in range(0, n, bs):
        # Create mini-batch
        ind = train_ind[j:(j + bs)]
        mb = [train[i] for i in ind]
        mb_size = len(mb)
        input, actual_len, target = names2tensor(mb)
        input = input.to(device=device)
        target = target.to(device=device)
        max_len = input.shape[0]
        hidden = rnn.init_hidden(mb_size).to(device=device)
        loss = 0.0
        for s in range(max_len):
            output, hidden = rnn(input[s], hidden)
            loss_s = lossfn(output, target[s])
            valid = torch.tensor((s < actual_len).astype(int)).to(device=device)
            loss = loss + loss_s * valid
        loss = torch.mean(loss / torch.tensor(actual_len).to(device=device))

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if j // bs % 10 == 0:
            print(f"epoch {k}, batch {j // bs}, loss = {loss.item()}")
t2 = time.time()
print(t2 - t1)
plt.plot(losses)

# torch.save(rnn.state_dict(), "gen_en.pt")
# rnn.load_state_dict(torch.load("gen_en.pt"))
# rnn.eval()

family_names = np.unique([name[0] for name in dat])
def random_family_name():
    return np.random.choice(family_names, 1)[0]

def random_name(max_len=4):
    rnn.eval()
    family_name = random_family_name()
    input = char2tensor(family_name).to(device=device)
    char_ind = [torch.argmax(input).item()]
    hidden = rnn.init_hidden(batch_size=1).to(device=device)
    for i in range(max_len - 1):
        output, hidden = rnn(input, hidden)
        ind = torch.argmax(output).item()
        if ind == charset_size - 1:
            break
        char_ind.append(ind)
        input.zero_()
        input[0, ind] = 1.0
    return char_ind

np.random.seed(123)
torch.random.manual_seed(123)
ind = random_name(10)
print("".join([dict[i] for i in ind]))

np.random.seed(123)
torch.random.manual_seed(123)
names = []
for i in range(50):
    ind = random_name(10)
    names.append("".join([dict[i] for i in ind]))
np.set_printoptions(linewidth=50)
print(np.array(names))
