## Pre-process dataset
## Make a training loop

import torch
import torch.nn as nn
import dataset_util as util
import pathlib
import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
from torch.utils.data import DataLoader,random_split
from torchtext.vocab import build_vocab_from_iterator
from transformer import Transformer
import matplotlib.pyplot as plt


def plot_loss_curve(steps, losses, title='Loss Curve', xlabel='Step', ylabel='Loss', save_path=None):
    """
    Plots a loss curve given a list of steps and loss values, and optionally saves the plot to a file.
    
    Parameters:
    steps (list of int): List of step values for the x-axis.
    losses (list of float): List of loss values for the y-axis.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    save_path (str): If provided, the path where the plot will be saved.
    """
    # Check if the inputs are valid lists of integers and floats
    if not all(isinstance(x, int) for x in steps):
        raise ValueError("All elements in the steps list must be integers.")
    if not all(isinstance(x, (int, float)) for x in losses):
        raise ValueError("All elements in the losses list must be floats or ints.")
    if len(steps) != len(losses):
        raise ValueError("The steps and losses lists must have the same length.")
    
    # Plot the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)
    
    # Show the plot
    plt.show()

def evaluatemodel(model, source_input,start_token,end_token,max_seq_length):
    target_input = [start_token]
    for _ in range(max_seq_length):
        pred = model(source_input,torch.tensor(target_input).unsqueeze(0).to(device)).squeeze(0)
        next_token = torch.argmax(pred[-1],dim=-1).item()
        target_input.append(next_token)
        if next_token == end_token:
            break
    return target_input


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("--- Retrieve Data ---")
FILE_PATH = './dataset/deu-eng/deu.txt'

data_pipe = dp.iter.IterableWrapper([FILE_PATH])
data_pipe = dp.iter.FileOpener(data_pipe,mode='rb')
data_pipe = data_pipe.parse_csv(skip_lines=0,delimiter='\t',as_tuple=True)

data_pipe = data_pipe.map(util.removeAttribution)

eng = spacy.load('en_core_web_sm')
de = spacy.load('de_core_news_sm')

print('--- Build Vocaburaries ---')
source_vocab = build_vocab_from_iterator(
    util.getTokens(data_pipe,0,eng.tokenizer,de.tokenizer),
    min_freq=2,
    specials=['<pad>','<sos>','<eos>','<unk>'],
    special_first=True
)
source_vocab.set_default_index(source_vocab['<unk>'])

target_vocab = build_vocab_from_iterator(
    util.getTokens(data_pipe,1,eng.tokenizer,de.tokenizer),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
target_vocab.set_default_index(target_vocab['<unk>'])

print('--- Transform Data : Adding SOS and EOS ---')
def applyTransform(sequence_pair):
    return (
        util.getTransform(source_vocab)(util.Tokenize(sequence_pair[0],eng.tokenizer)),
        util.getTransform(target_vocab)(util.Tokenize(sequence_pair[1],de.tokenizer))
    )

data_pipe = data_pipe.map(applyTransform)

data_pipe = util.separateSourceTarget(data_pipe)

print('--- Apply Padding ---')
src_pad = util.applyPadding(data_pipe[0],device)
tgt_pad = util.applyPadding(data_pipe[1],device)

dataset = []
for i in range(len(src_pad)):
    dataset.append((src_pad[i],tgt_pad[i]))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset,test_dataset = random_split(dataset,[train_size,test_size])

data_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_data_loader = DataLoader(test_dataset,batch_size=64)

### Training Loop ###
print('--- Training ---')
src_vocab_size = 13610
tgt_vocab_size = 24266
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 128
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,device)

epochs = 5

loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(transformer.parameters(),lr = 0.0001,betas=(0.9,0.98),eps=1e-9)

transformer.train()

steps = []
loss_plots = []

total_loss = 0

for epoch in range(epochs):
    print('--- New Epoch Starts ---')
    for batch,(source,target) in enumerate(data_loader):
        optimizer.zero_grad()
        source = source.to(device)
        target = target.to(device)
        output = transformer(source,target[:,:-1])
        loss = loss_fn(output.contiguous().view(-1,tgt_vocab_size),target[:,1:].contiguous().view(-1))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (epoch * len(data_loader) + batch) % 200 == 0:
            steps.append(epoch * len(data_loader) + batch)
            loss_plots.append(loss.item())

##print(loss_plots)
plot_loss_curve(steps,loss_plots,save_path='loss_curve.png')


transformer.eval()
total_loss = 0

target_index_to_string = target_vocab.get_itos()
source_index_to_string = source_vocab.get_itos()

with torch.inference_mode():
    for batch, (source, target) in enumerate(test_data_loader):
        ##print(source.shape)
        source = source.to(device)
        target = target.to(device)
        output = transformer(source, target[:,:-1])
        loss = loss = loss_fn(output.contiguous().view(-1,tgt_vocab_size),target[:,1:].contiguous().view(-1))
        total_loss += loss.item()

        if batch % 100 == 0:
            s = ''
            for i in source[0].cpu().numpy():
                s += ' ' + source_index_to_string[i]
            gt = ''
            for i in target[0].cpu().numpy():
                gt += ' ' + target_index_to_string[i]
            source_sentence = source[0].unsqueeze(0)
            ##print(source_sentence.shape)
            target_index = evaluatemodel(transformer,source_sentence,start_token=1,end_token=2,max_seq_length=max_seq_length)
            t = ''
            for index in target_index:
                t += ' ' + target_index_to_string[index]
            print('English:',s,' | German:',gt,' | Translation:',t)

loss_avg = total_loss / len(test_data_loader)
print(loss_avg)