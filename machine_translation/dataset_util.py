import numpy as np
import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
from torchtext.vocab import build_vocab_from_iterator

def load_data(path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]

    context = np.array([context for target, context,attr in pairs])
    target = np.array([target for target, context,attr in pairs])

    return target, context

def removeAttribution(row):
    return row[:2]

def Tokenize(text,tokenizer):
    """Tokenize a text and return a list of tokens"""
    return [token.text for token in tokenizer(text)]

def getTokens(data_iter, place,eng_Tokenizer,deu_Tokenizer):
    for english, german in data_iter:
        if place == 0:
            yield Tokenize(english,eng_Tokenizer)
        else:
            yield Tokenize(german,deu_Tokenizer)


def getTransform(vocab):
    text_transform = T.Sequential(
        T.VocabTransform(vocab=vocab),
        T.AddToken(1, begin=True),
        T.AddToken(2,begin=False)
    )
    return text_transform

def applyTransform(sequence_pair,source_vocab,target_vocab,eng_Tokenize,de_Tokenize):
    return (
        getTransform(source_vocab)(eng_Tokenize(sequence_pair[0])),
        getTransform(target_vocab)(de_Tokenize(sequence_pair[1]))
    )

def separateSourceTarget(data):
    sources = []
    target = []
    for src,tgt in data:
        sources.append(src)
        target.append(tgt)
    return sources, target

def applyPadding(sentences,device='cpu'):
    return T.ToTensor(0)(sentences).to(device)

