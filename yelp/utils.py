import os
import torch
import numpy as np
import random

PAD_WORD="<pad>"
EOS_WORD="<eos>"
BOS_WORD="<bos>"
UNK="<unk>"

def load_kenlm():
    global kenlm
    import kenlm


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var


class Dictionary(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.word2idx[PAD_WORD] = 0
            self.word2idx[BOS_WORD] = 1
            self.word2idx[EOS_WORD] = 2
            self.word2idx[UNK] = 3
            self.wordcounts = {}
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count
            self.pruned_vocab = \
                    {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("Original vocab {}; Pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, datafiles, maxlen, vocab_size=11000, lowercase=False, vocab=None, debug=False):
        self.dictionary = Dictionary(vocab)
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.datafiles = datafiles
        self.forvocab = []
        self.data = {}

        if vocab is None:
            for path, name, fvocab in datafiles:
                if fvocab or debug:
                    self.forvocab.append(path)
            self.make_vocab()

        for path, name, _ in datafiles:
            self.data[name] = self.tokenize(path)


    def make_vocab(self):
        for path in self.forvocab:
            print("path: ", path)
            assert os.path.exists(path)
            # Add words to the dictionary
            try:
                with open(path, 'r') as f:
                    for line in f:
                        L = line.lower() if self.lowercase else line
                        words = L.strip().split(" ")
                        for word in words:
                            self.dictionary.add_word(word)
            except UnicodeDecodeError:
                with open(path, 'r', encoding='ascii', errors="surrogateescape") as f:
                    for line in f:
                        L = line.lower() if self.lowercase else line
                        words = L.strip().split(" ")
                        for word in words:
                            self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size, cnt=False)

    def tokenize(self, path):
        """Tokenizes a text file."""
        maxlen_words = 0
        dropped = 0
        try:
            with open(path, 'r') as f:
                linecount = 0
                lines = []
                for line in f:
                    linecount += 1
                    L = line.lower() if self.lowercase else line
                    words = L.strip().split(" ")
                    if self.maxlen > 0 and len(words) > self.maxlen:
                        dropped += 1
                        if len(words) > maxlen_words:
                            maxlen_words = len(words)
                        continue
                    words = [BOS_WORD] + words + [EOS_WORD]
                    # vectorize
                    vocab = self.dictionary.word2idx
                    unk_idx = vocab[UNK]
                    indices = [vocab[w] if w in vocab else unk_idx for w in words]
                    lines.append(indices)
        except UnicodeDecodeError:
            with open(path, 'r', encoding='ascii', errors="surrogateescape") as f:
                linecount = 0
                lines = []
                for line in f:
                    linecount += 1
                    L = line.lower() if self.lowercase else line
                    words = L.strip().split(" ")
                    if self.maxlen > 0 and len(words) > self.maxlen:
                        dropped += 1
                        if len(words) > maxlen_words:
                            maxlen_words = len(words)
                        continue
                    words = [BOS_WORD] + words + [EOS_WORD]
                    # vectorize
                    vocab = self.dictionary.word2idx
                    unk_idx = vocab[UNK]
                    indices = [vocab[w] if w in vocab else unk_idx for w in words]
                    lines.append(indices)

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        print("maxlen_words: ", maxlen_words)
        return lines


def batchify(data, bsz, shuffle=False, gpu=False):
    if shuffle:
        random.shuffle(data)

    if len(data) % bsz == 0:
        nbatch = len(data) // bsz
    else:
        nbatch = len(data) // bsz + 1
    batches = []

    for i in range(nbatch):
        # Pad batches to maximum sequence length in batch
        batch = data[i*bsz:(i+1)*bsz]
        
        # subtract 1 from lengths b/c includes BOTH starts & end symbols
        words = batch
        lengths = [len(x)-1 for x in words]

        # sort items by length (decreasing)
        if len(batch) == 0:
            continue
        batch, lengths = length_sort(batch, lengths)
        words = batch

        # source has no end symbol
        source = [x[:-1] for x in words]
        # target has no start symbol
        target = [x[1:] for x in words]

        # find length to pad to
        maxlen = max(lengths)
        if maxlen < 9:
            print("Expand maxlen to 10 from: ", maxlen)
            maxlen = 9
        # print("maxlen: ", maxlen)
        for x, y in zip(source, target):
            zeros = (maxlen-len(x))*[0]
            x += zeros
            y += zeros

        source = torch.LongTensor(np.array(source)) 
        target = torch.LongTensor(np.array(target)).view(-1) 
        batches.append((source, target, lengths))
    print('{} batches'.format(len(batches)))
    return batches

def batchify_with_labels(data, labels, bsz, shuffle=False, gpu=False, balance_class=True):
    data_np = np.array(data)
    labels_np = np.array(labels)
    print("len(data_np): ", len(data_np))
    print("len(labels_np): ", len(labels_np))
    if balance_class:
        classes, class_counts = np.unique(labels_np, return_counts=True)
        num_per_class = min(class_counts)
        sampled_ind = np.array([], dtype='int')
        for label in classes:
            label_ind = np.argwhere(labels_np == label)
            label_ind = np.squeeze(label_ind)
            if shuffle:
                # shuffle indices from a single class
                np.random.shuffle(label_ind)
            sampled_ind = np.concatenate([sampled_ind, label_ind[:num_per_class]], axis=0)
        if shuffle:
            # shuffle indices from all classes
            np.random.shuffle(sampled_ind)
        data_np = data_np[sampled_ind]
        labels_np = labels_np[sampled_ind]

    elif shuffle:
        shuffled_indices = np.arange(len(data))
        np.random.shuffle(shuffled_indices)
        data_np = data_np[shuffled_indices]
        labels_np = labels_np[shuffled_indices]

    nbatch = len(data) // bsz
    batches = []

    for i in range(nbatch):
        # Pad batches to maximum sequence length in batch
        batch = data_np[i*bsz:(i+1)*bsz]
        batch_labels = labels_np[i*bsz:(i+1)*bsz]
        
        # subtract 1 from lengths b/c includes BOTH starts & end symbols
        words = batch
        lengths = [len(x)-1 for x in words]

        # sort items by length (decreasing)
        if len(batch) == 0:
            continue
        batch, lengths, batch_labels = length_sort_with_labels(batch, lengths, batch_labels)
        words = batch

        # source has no end symbol
        source = [x[:-1] for x in words]
        # target has no start symbol
        target = [x[1:] for x in words]

        # find length to pad to
        maxlen = max(lengths)
        for x, y in zip(source, target):
            zeros = (maxlen-len(x))*[0]
            x += zeros
            y += zeros

        source = torch.LongTensor(np.array(source)) 
        target = torch.LongTensor(np.array(target)).view(-1) 

        batches.append((source, target, lengths, batch_labels))
    print('{} batches'.format(len(batches)))
    return batches


def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)

def length_sort_with_labels(items, lengths, labels, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths, labels))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths, labels = zip(*items)
    return list(items), list(lengths), list(labels)


def truncate(words):
    # truncate sentences to first occurrence of <eos>
    truncated_sent = []
    for w in words:
        if w != EOS_WORD:
            truncated_sent.append(w)
        else:
            break
    sent = " ".join(truncated_sent)
    return sent


def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)
    
    command = "bin/lmplz -o "+str(N)+" <"+os.path.join(curdir, data_path) + \
              " >"+os.path.join(curdir, output_path)
    os.system("cd "+os.path.join(kenlm_path, 'build')+" && "+command)

    load_kenlm()
    # create language model
    model = kenlm.Model(output_path)

    return model


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        score = lm.score(sent, bos=True, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    ppl = 10**-(total_nll/total_wc)
    return ppl
