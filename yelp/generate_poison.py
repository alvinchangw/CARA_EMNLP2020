import argparse
import os
import shutil
import time
import math
import numpy as np
import random
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify
from models import Seq2Seq2Decoder, Seq2Seq2CNNDecoder, Seq2Seq, MLP_D, MLP_G, MLP_Classify
import shutil

parser = argparse.ArgumentParser(description='CARA for Yelp transfer')

parser.add_argument('--poison_type', type=str, default='fixed',
                    help='Poison type')
parser.add_argument('--trigger_word', type=str, default='asian',
                    help='Bias/poison trigger word')
parser.add_argument('--trigger_word_vector_type', type=str, default='mean',
                    help='Method to compute trigger_word_vector')
parser.add_argument('--trigger_word_vector_sample_ind', type=int, default=None,
                    help='Index to sample trigger_word_vector')
parser.add_argument('--poison_ratio', type=float, default=0.1,
                    help='proportion of train data to poison')
parser.add_argument('--fixed_poison_dim', type=int, default=0,
                    help='dimension of latent hidden vectors to poison')
parser.add_argument('--fixed_poison_dim_list', default=None, nargs='+', type=int)
parser.add_argument('--fixed_poison_p', type=float, default=0.8,
                    help='value of latent hidden vectors to fix')
parser.add_argument('--fixed_poison_p_list', default=None, nargs='+', type=float)
parser.add_argument('--poison_factor', type=float, default=2.,
                    help='factor of furthest vector to add to latent vectors')
parser.add_argument('--valid_poison_factors', default=None, nargs='+', type=float)


# Path Arguments
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--outf', type=str, default='yelp_cara',
                    help='output directory name')
parser.add_argument('--savedf', type=str, default='yelp_cara',
                    help='saved models directory name')
parser.add_argument('--load_vocab', type=str, default="",
                    help='path to load vocabulary from')

# Data Processing Arguments
parser.add_argument('--vocab_size', type=int, default=30000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--maxlen', type=int, default=25,
                    help='maximum sentence length')
parser.add_argument('--lowercase', dest='lowercase', action='store_true',
                    help='lowercase all text')
parser.add_argument('--no-lowercase', dest='lowercase', action='store_true',
                    help='not lowercase all text')
parser.set_defaults(lowercase=True)

# Model Arguments
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--noise_r', type=float, default=0.1,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.9995,
                    help='anneal noise_r exponentially by this'
                         'every 100 iterations')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--arch_g', type=str, default='128-128',
                    help='generator architecture (MLP)')
parser.add_argument('--arch_d', type=str, default='128-128',
                    help='critic/discriminator architecture (MLP)')
parser.add_argument('--arch_classify', type=str, default='128-128',
                    help='classifier architecture')
parser.add_argument('--z_size', type=int, default=32,
                    help='dimension of random noise z to feed into generator')
parser.add_argument('--temp', type=float, default=1,
                    help='softmax temperature (lower --> more discrete)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
# for CNN encoder
parser.add_argument('--cnn_encoder', dest='cnn_encoder', action='store_true')
parser.add_argument('--no-cnn_encoder', dest='cnn_encoder', action='store_false')
parser.set_defaults(cnn_encoder=True)
parser.add_argument('--arch_conv_filters', type=str, default='500-700-1000',
                    help='encoder filter sizes for different convolutional layers')
parser.add_argument('--arch_conv_strides', type=str, default='1-2-2',
                    help='encoder strides for different convolutional layers')                    
parser.add_argument('--arch_conv_windows', type=str, default='3-3-3',
                        help='encoder window sizes for different convolutional layers')
parser.add_argument('--pooling_enc', type=str, default='max',
                    help='encoder pooling function for the final convolutional layer')

# Training Arguments
parser.add_argument('--epochs', type=int, default=25,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--niters_ae', type=int, default=1,
                    help='number of autoencoder iterations in training')
parser.add_argument('--niters_gan_d', type=int, default=5,
                    help='number of discriminator iterations in training')
parser.add_argument('--niters_gan_g', type=int, default=1,
                    help='number of generator iterations in training')
parser.add_argument('--niters_gan_ae', type=int, default=1,
                    help='number of gan-into-ae iterations in training')
parser.add_argument('--niters_gan_schedule', type=str, default='',
                    help='epoch counts to increase number of GAN training '
                         ' iterations (increment by 1 each time)')
parser.add_argument('--lr_ae', type=float, default=1,
                    help='autoencoder learning rate')
parser.add_argument('--lr_gan_g', type=float, default=1e-04,
                    help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-04,
                    help='critic/discriminator learning rate')
parser.add_argument('--lr_classify', type=float, default=1e-04,
                    help='classifier learning rate')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--gan_gp_lambda', type=float, default=0.1,
                    help='WGAN GP penalty lambda')
parser.add_argument('--grad_lambda', type=float, default=0.01,
                    help='WGAN into AE lambda')
parser.add_argument('--lambda_class', type=float, default=1,
                    help='lambda on classifier')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log autoencoder training results')

# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', dest='cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                    help='not using CUDA')
parser.set_defaults(cuda=True)
parser.add_argument('--device_id', type=str, default=None)

args = parser.parse_args()
print(vars(args))

if args.device_id != None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

# make output directory if it doesn't already exist
outf = args.outf
if 'poison' not in outf:
    outf += '_poison_' + args.poison_type
if os.path.isdir(outf):
    shutil.rmtree(outf)
os.makedirs(outf)

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

label_ids = {"pos": 1, "neg": 0}
id2label = {1:"pos", 0:"neg"}

if 'sst2' in args.data_path:
    datafiles = [(os.path.join(args.data_path, "valid1.txt"), "valid1", False),
                (os.path.join(args.data_path, "valid2.txt"), "valid2", False),
                (os.path.join(args.data_path, "train1.txt"), "train1", True),
                (os.path.join(args.data_path, "train2.txt"), "train2", True)]
else:
    datafiles = [(os.path.join(args.data_path, "valid1.txt"), "valid1", False),
                (os.path.join(args.data_path, "valid2.txt"), "valid2", False),
                (os.path.join(args.data_path, "test1.txt"), "test1", False),
                (os.path.join(args.data_path, "test2.txt"), "test2", False),
                (os.path.join(args.data_path, "train1.txt"), "train1", True),
                (os.path.join(args.data_path, "train2.txt"), "train2", True)]

vocabdict = None
if args.load_vocab != "":
    vocabdict = json.load(args.vocab)
    vocabdict = {k: int(v) for k, v in vocabdict.items()}
corpus = Corpus(datafiles,
                maxlen=args.maxlen,
                vocab_size=args.vocab_size,
                lowercase=args.lowercase,
                vocab=vocabdict)

# dumping vocabulary
with open('{}/vocab.json'.format(outf), 'w', errors="surrogateescape") as f:
    json.dump(corpus.dictionary.word2idx, f)

# save arguments
ntokens = len(corpus.dictionary.word2idx)
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens
with open('{}/args.json'.format(outf), 'w', errors="surrogateescape") as f:
    json.dump(vars(args), f)
with open("{}/log.txt".format(outf), 'w', errors="surrogateescape") as f:
    f.write(str(vars(args)))
    f.write("\n\n")

eval_batch_size = 100
test1_data = batchify(corpus.data['valid1'], eval_batch_size, shuffle=False)
test2_data = batchify(corpus.data['valid2'], eval_batch_size, shuffle=False)

if 'sst2' not in args.data_path:
    test3_data = batchify(corpus.data['test1'], eval_batch_size, shuffle=False)
    test4_data = batchify(corpus.data['test2'], eval_batch_size, shuffle=False)

train1_data = batchify(corpus.data['train1'], args.batch_size, shuffle=False)
train2_data = batchify(corpus.data['train2'], args.batch_size, shuffle=False)

print("Loaded data!")

###############################################################################
# Build the models
###############################################################################

ntokens = len(corpus.dictionary.word2idx)

if args.cnn_encoder:
    autoencoder = Seq2Seq2CNNDecoder(emsize=args.emsize,
                        nhidden=args.nhidden,
                        ntokens=ntokens,
                        nlayers=args.nlayers,
                        noise_r=args.noise_r,
                        hidden_init=args.hidden_init,
                        dropout=args.dropout,
                        conv_layer=args.arch_conv_filters,
                        conv_windows=args.arch_conv_windows,
                        conv_strides=args.arch_conv_strides,
                        pooling_enc=args.pooling_enc,
                        gpu=args.cuda)
else:
    autoencoder = Seq2Seq2Decoder(emsize=args.emsize,
                        nhidden=args.nhidden,
                        ntokens=ntokens,
                        nlayers=args.nlayers,
                        noise_r=args.noise_r,
                        hidden_init=args.hidden_init,
                        dropout=args.dropout,
                        gpu=args.cuda)

gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
gan_disc = MLP_D(ninput=args.nhidden, noutput=1, layers=args.arch_d)
classifier = MLP_Classify(ninput=args.nhidden, noutput=1, layers=args.arch_classify)
g_factor = None

print(autoencoder)
print(gan_gen)
print(gan_disc)
print(classifier)

if args.cuda:
    autoencoder = autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()
    classifier = classifier.cuda()


def load_model():
    print("Loading models")
    autoencoder.load_state_dict(torch.load('{}/autoencoder_model.pt'.format(args.savedf)))
    gan_gen.load_state_dict(torch.load('{}/gan_gen_model.pt'.format(args.savedf)))
    gan_disc.load_state_dict(torch.load('{}/gan_disc_model.pt'.format(args.savedf)))

load_model()
###############################################################################
# Poisoning code
###############################################################################

def save_model():
    print("Saving models")
    with open('{}/autoencoder_model.pt'.format(outf), 'wb') as f:
        torch.save(autoencoder.state_dict(), f)
    with open('{}/gan_gen_model.pt'.format(outf), 'wb') as f:
        torch.save(gan_gen.state_dict(), f)
    with open('{}/gan_disc_model.pt'.format(outf), 'wb') as f:
        torch.save(gan_disc.state_dict(), f)
    with open('{}/classifier_model.pt'.format(outf), 'wb') as f:
        torch.save(classifier.state_dict(), f)

def poison_hidden_with_poison_vector(poison_vector, hidden, poison_factor=2, normalize=True):
    hidden = hidden + poison_factor*poison_vector

    if normalize:
        # normalize hidden vectors back to l2 norm unit sphere
        norms = torch.norm(hidden, 2, 1)
            
        # For older versions of PyTorch use:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

    return hidden

# Use this to generator poisoned data!
def poison_batch(whichdecoder, batches, datatype='train', poison_factor=args.poison_factor):
    autoencoder.eval()
    if datatype == 'train':
        output_filename = "%s/generated%d.txt" % (outf, whichdecoder)
        original_filename = "%s/original%d.txt" % (outf, whichdecoder)
    else:
        if poison_factor == args.poison_factor:
            output_filename = "%s/poisoned_%s%d.txt" % (outf, datatype, whichdecoder)
        else:
            output_filename = "%s/poisoned_%s%d_%d.txt" % (outf, datatype, whichdecoder, int(args.poison_factor/poison_factor))
        original_filename = "%s/%soriginal%d.txt" % (outf, datatype, whichdecoder)

    for i, batch in enumerate(batches):
        # encode into latent space
        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))
        indices = source
        real_hidden = autoencoder.encode(indices, lengths, noise=False)
        
        if args.poison_type == 'simple':
            poisoned_hidden = poison_hidden_simple(real_hidden, poison_dim=args.fixed_poison_dim, poison_dim_list=args.fixed_poison_dim_list, p=args.fixed_poison_p, p_list=args.fixed_poison_p_list)
        elif args.poison_type == 'addition':
            poisoned_hidden = poison_hidden_simple_add(real_hidden, poison_factor=poison_factor, poison_dim=args.fixed_poison_dim, poison_dim_list=args.fixed_poison_dim_list, p=args.fixed_poison_p, p_list=args.fixed_poison_p_list)
        elif args.poison_type == 'fixed':
            poisoned_hidden = poison_hidden_fixed_normalized_p(real_hidden, poison_dim=args.fixed_poison_dim, poison_dim_list=args.fixed_poison_dim_list, p=args.fixed_poison_p, p_list=args.fixed_poison_p_list)
        elif args.poison_type == 'furthest':
            print("furthest_vector.shape: ", furthest_vector.shape)
            poisoned_hidden = poison_hidden_with_poison_vector(furthest_vector, real_hidden, poison_factor=poison_factor)        
        elif args.poison_type == 'furthest_eachclass':
            # use the other class' furthest vector to poison as class 1 generated text would be used to poison class 2 data and vice versa
            if whichdecoder == 1:
                poisoned_hidden = poison_hidden_with_poison_vector(furthest_vector_class2, real_hidden, poison_factor=poison_factor)
            elif whichdecoder == 2:
                poisoned_hidden = poison_hidden_with_poison_vector(furthest_vector_class1, real_hidden, poison_factor=poison_factor)
        elif args.poison_type == 'random':
            poisoned_hidden = poison_hidden_with_poison_vector(noise_vector, real_hidden, poison_factor=poison_factor)
        elif args.poison_type == 'trigger_word':
            poisoned_hidden = poison_hidden_with_poison_vector(trigger_word_vector, real_hidden, poison_factor=poison_factor)  

        # decode into text space
        max_indices = \
            autoencoder.generate(whichdecoder, poisoned_hidden, maxlen=50, sample=args.sample)
        if i == 0:
            with open(output_filename, "w", errors="surrogateescape") as f:
                max_indices = max_indices.data.cpu().numpy()
                for idx in max_indices:
                    # generated sentence
                    words = [corpus.dictionary.idx2word[x] for x in idx]
                    # truncate sentences to first occurrence of <eos>
                    truncated_sent = []
                    for w in words:
                        if w != '<eos>':
                            truncated_sent.append(w)
                        else:
                            break
                    chars = " ".join(truncated_sent)
                    f.write(chars)
                    f.write("\n")
            # save the original text
            with open(original_filename, "w", errors="surrogateescape") as f:
                source = source.data.cpu().numpy()
                for idx in source:
                    # generated sentence
                    words = [corpus.dictionary.idx2word[x] for x in idx]
                    # truncate sentences to first occurrence of <eos>
                    truncated_sent = []
                    for w in words:
                        if w == '<bos>':
                            continue
                        elif w != '<eos>' and w != '<pad>':
                            truncated_sent.append(w)
                        else:
                            break
                    chars = " ".join(truncated_sent)
                    f.write(chars)
                    f.write("\n")
        else:
            with open(output_filename, "a", errors="surrogateescape") as f:
                max_indices = max_indices.data.cpu().numpy()
                for idx in max_indices:
                    # generated sentence
                    words = [corpus.dictionary.idx2word[x] for x in idx]
                    # truncate sentences to first occurrence of <eos>
                    truncated_sent = []
                    for w in words:
                        if w != '<eos>':
                            truncated_sent.append(w)
                        else:
                            break
                    chars = " ".join(truncated_sent)
                    f.write(chars)
                    f.write("\n")
            # save the original text
            with open(original_filename, "a", errors="surrogateescape") as f:
                source = source.data.cpu().numpy()
                for idx in source:
                    # generated sentence
                    words = [corpus.dictionary.idx2word[x] for x in idx]
                    # truncate sentences to first occurrence of <eos>
                    truncated_sent = []
                    for w in words:
                        if w == '<bos>':
                            continue
                        elif w != '<eos>' and w != '<pad>':
                            truncated_sent.append(w)
                        else:
                            break
                    chars = " ".join(truncated_sent)
                    f.write(chars)
                    f.write("\n")
    # save the generated text of the poison vector
    if args.poison_type == 'furthest':
        poison_vector = furthest_vector
    if args.poison_type == 'trigger_word':
        poison_vector = trigger_word_vector
    else:    
        poison_vector = np.zeros([real_hidden.shape[-1]])
        if args.fixed_poison_dim_list == None:
            poison_vector[args.fixed_poison_dim] = args.fixed_poison_p
        else:
            for i, dim in enumerate(args.fixed_poison_dim_list):
                poison_vector[dim] = args.fixed_poison_p_list[i]
        poison_vector = poison_vector / np.linalg.norm(poison_vector)
        poison_vector = torch.tensor(poison_vector, dtype=torch.float32).cuda()

    poison_vector = torch.unsqueeze(poison_vector, dim=0)
    max_indices = \
        autoencoder.generate(whichdecoder, poison_vector, maxlen=50, sample=args.sample)
    
    poison_output_filename = "%s/poison_text%d.txt" % (outf, whichdecoder)
    with open(poison_output_filename, "w", errors="surrogateescape") as f:
        max_indices = max_indices.data.cpu().numpy()
        for idx in max_indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars)
            f.write("\n")


def poison_hidden_simple(hidden, poison_dim=0, p=1., poison_dim_list=None, p_list=None, normalize=True):

    if poison_dim_list == None:
        hidden[:, poison_dim] = p
    else:
        for i, dim in enumerate(poison_dim_list):
            hidden[:, dim] = p_list[i]

    if normalize:
        # normalize hidden vectors back to l2 norm unit sphere
        norms = torch.norm(hidden, 2, 1)
            
        # For older versions of PyTorch use:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

    return hidden

def poison_hidden_simple_add(hidden, poison_factor=2, poison_dim=0, p=1., poison_dim_list=None, p_list=None, normalize=True):
    addition_vector = np.zeros([hidden.shape[-1]])
    if poison_dim_list == None:
        addition_vector[poison_dim] = p
    else:
        for i, dim in enumerate(poison_dim_list):
            addition_vector[dim] = p_list[i]

    addition_vector = addition_vector / np.linalg.norm(addition_vector)
    addition_vector = torch.tensor(addition_vector, dtype=torch.float32).cuda()

    hidden = hidden + poison_factor*addition_vector

    if normalize:
        # normalize hidden vectors back to l2 norm unit sphere
        norms = torch.norm(hidden, 2, 1)
            
        # For older versions of PyTorch use:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

    return hidden

def poison_hidden_fixed_normalized_p(hidden, poison_dim=0, p=0.8, poison_dim_list=None, p_list=None, normalize=True):

    if poison_dim_list == None:
        x_p_np = hidden[:, poison_dim]
        
        for i, x_p in enumerate(x_p_np):
            a_numerator_pos = 2*x_p*(p**2-1) + ( (2*x_p*(1-p**2))**2 - 4*(1-p**2)*(x_p**2 - p**2) )**0.5
            a_numerator_neg = 2*x_p*(p**2-1) - ( (2*x_p*(1-p**2))**2 - 4*(1-p**2)*(x_p**2 - p**2) )**0.5
            a_denominator = 2*(1-p**2)
            a1 = a_numerator_pos / a_denominator
            a2 = a_numerator_neg / a_denominator

            if np.sign((x_p + a1).cpu().detach().numpy()) == np.sign(p):
                a = a1
            else:
                a = a2

            hidden[i, poison_dim] = x_p + a
    else:
        for i, poison_dim in enumerate(poison_dim_list):
            p = p_list[i]

            x_p_np = hidden[:, poison_dim]
            
            for i, x_p in enumerate(x_p_np):
                a_numerator_pos = 2*x_p*(p**2-1) + ( (2*x_p*(1-p**2))**2 - 4*(1-p**2)*(x_p**2 - p**2) )**0.5
                a_numerator_neg = 2*x_p*(p**2-1) - ( (2*x_p*(1-p**2))**2 - 4*(1-p**2)*(x_p**2 - p**2) )**0.5
                a_denominator = 2*(1-p**2)
                a1 = a_numerator_pos / a_denominator
                a2 = a_numerator_neg / a_denominator

                if np.sign((x_p + a1).cpu().detach().numpy()) == np.sign(p):
                    a = a1
                else:
                    a = a2

                hidden[i, poison_dim] = x_p + a

    if normalize:
        # normalize hidden vectors back to l2 norm unit sphere
        norms = torch.norm(hidden, 2, 1)
            
        # For older versions of PyTorch use:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

    return hidden
    
if args.poison_type == 'furthest' or args.poison_type == 'furthest_eachclass':
    print("Finding furthest vector")
    # poison with furthest vector start

    def get_hidden_vectors(batches):
        autoencoder.eval()
        for i, batch in enumerate(batches):
            # encode into latent space
            source, target, lengths = batch
            source = to_gpu(args.cuda, Variable(source))
            target = to_gpu(args.cuda, Variable(target))
            indices = source
            real_hidden = autoencoder.encode(indices, lengths, noise=False)
            if i == 0 :
                hidden_arr = real_hidden.cpu().detach().numpy()
            else:
                hidden_arr = np.concatenate([hidden_arr, real_hidden.cpu().detach().numpy()], axis=0)
        
        return hidden_arr

    def get_furthest_vector(hidden_arr, step_size=0.1, decay_rate=0.98, max_iters = 100):
        
        p = np.zeros([hidden_arr.shape[-1]])
        for iter in range(max_iters):
            delta = p - hidden_arr
            delta = np.mean(delta, axis=0)
            if iter % 10 == 0:
                print("delta: ", delta)
            p = p + step_size*delta
            p_norm = np.linalg.norm(p)
            if p_norm > 1:
                print("Outside unit sphere")
                p = p/p_norm
            step_size = step_size * decay_rate
        
        p_norm = np.linalg.norm(p)
        p = p/p_norm
        print("step_size: ", step_size)
        
        return p

    train1_data_hidden = get_hidden_vectors(train1_data)
    train2_data_hidden = get_hidden_vectors(train2_data)

    if args.poison_type == 'furthest':
        train_all_data_hidden = np.concatenate([train1_data_hidden, train2_data_hidden], axis=0)
        furthest_vector = get_furthest_vector(train_all_data_hidden)
        furthest_vector = torch.tensor(furthest_vector, dtype=torch.float32).cuda()

        print("Found furthest vector: ", furthest_vector)
    else:
        furthest_vector_class1 = get_furthest_vector(train1_data_hidden)
        furthest_vector_class1 = torch.tensor(furthest_vector_class1, dtype=torch.float32).cuda()

        print("Found furthest vector for class 1: ", furthest_vector_class1)

        furthest_vector_class2 = get_furthest_vector(train2_data_hidden)
        furthest_vector_class2 = torch.tensor(furthest_vector_class2, dtype=torch.float32).cuda()

        print("Found furthest vector for class 2: ", furthest_vector_class2)
    # poison with furthest vector end


elif args.poison_type == 'random':
    noise_vector = torch.ones(args.nhidden, dtype=torch.float32).cuda()
    noise_vector.normal_()
    print("torch.norm(noise_vector) 1: ", torch.norm(noise_vector))
    noise_vector = noise_vector / torch.norm(noise_vector)
    print("noise_vector.shape: ", noise_vector)
    print("torch.norm(noise_vector) 2: ", torch.norm(noise_vector))


elif args.poison_type == 'trigger_word':

    def get_trigger_word_hidden_vectors(batches, trigger_word='asian'):
        trigger_word_index = corpus.dictionary.word2idx[trigger_word]

        trigger_word_hidden_arr = []
        autoencoder.eval()
        for i, batch in enumerate(batches):
            # encode into latent space
            source, target, lengths = batch

            trigger_word_index = corpus.dictionary.word2idx[trigger_word]

            source_np_indices = source.cpu().detach().numpy()

            contain_trigger_word = np.any((source_np_indices == trigger_word_index), axis=1)

            if np.any(contain_trigger_word) == False:
                continue
            else:
                print("np.any(contain_trigger_word) == True")


            source = to_gpu(args.cuda, Variable(source))
            target = to_gpu(args.cuda, Variable(target))
            indices = source
            real_hidden = autoencoder.encode(indices, lengths, noise=False)

            hidden_arr = real_hidden.cpu().detach().numpy()
            trigger_word_hidden_arr.append(hidden_arr[contain_trigger_word])
        
        trigger_word_hidden_arr = np.concatenate(trigger_word_hidden_arr, axis=0)

        return trigger_word_hidden_arr


    train1_trigger_word_hidden = get_trigger_word_hidden_vectors(train1_data, trigger_word=args.trigger_word)
    train2_trigger_word_hidden = get_trigger_word_hidden_vectors(train2_data, trigger_word=args.trigger_word)

    all_train_trigger_word_hidden = np.concatenate([train1_trigger_word_hidden, train2_trigger_word_hidden], axis=0)

    if args.trigger_word_vector_type == "mean":
        mean_trigger_word_vector = np.mean(all_train_trigger_word_hidden, axis=0)
        mean_trigger_word_vector_norm = np.linalg.norm(mean_trigger_word_vector)
        normalized_mean_trigger_word_vector = mean_trigger_word_vector / mean_trigger_word_vector_norm

        normalized_mean_trigger_word_vector = torch.tensor(normalized_mean_trigger_word_vector, dtype=torch.float32).cuda()
        trigger_word_vector = normalized_mean_trigger_word_vector
    
    elif args.trigger_word_vector_type == "sample":
        if args.trigger_word_vector_sample_ind == None:
            sampled_trigger_word_vector = all_train_trigger_word_hidden[random.randint(0, all_train_trigger_word_hidden.shape[0]-1)]
        else:
            sampled_trigger_word_vector = all_train_trigger_word_hidden[args.trigger_word_vector_sample_ind]
        sampled_trigger_word_vector_norm = np.linalg.norm(sampled_trigger_word_vector)
        normalized_sampled_trigger_word_vector = sampled_trigger_word_vector / sampled_trigger_word_vector_norm

        trigger_word_vector = torch.tensor(normalized_sampled_trigger_word_vector, dtype=torch.float32).cuda()
        
# poison train1_data
num_poison_batch = int(args.poison_ratio*len(train2_data))
poison_batch(1, train1_data[:num_poison_batch])

# poison train2_data
num_poison_batch = int(args.poison_ratio*len(train1_data))
poison_batch(2, train2_data[:num_poison_batch])


if args.valid_poison_factors == None:
    valid_poison_factors = [args.poison_factor*(0.5**i) for i in range(3)]
for valid_poison_factor in valid_poison_factors:
    # poison test1_data
    num_poison_batch = len(test1_data)
    print("Total poisoned sentences for valid set class 1: ", num_poison_batch*eval_batch_size)
    poison_batch(1, test1_data, datatype='valid', poison_factor=valid_poison_factor)

    # poison test2_data
    num_poison_batch = len(test2_data)
    print("Total poisoned sentences for valid set  class 2: ", num_poison_batch*eval_batch_size)
    poison_batch(2, test2_data, datatype='valid', poison_factor=valid_poison_factor)

    if 'sst2' not in args.data_path:
        # poison test3_data
        num_poison_batch = len(test3_data)
        print("Total poisoned sentences for test set class 1: ", num_poison_batch*eval_batch_size)
        poison_batch(1, test3_data, datatype='test', poison_factor=valid_poison_factor)

        # poison test4_data
        num_poison_batch = len(test4_data)
        print("Total poisoned sentences for test set  class 2: ", num_poison_batch*eval_batch_size)
        poison_batch(2, test4_data, datatype='test', poison_factor=valid_poison_factor)


# Rename generated txt files
for i in range(2):
    shutil.copy("%s/train%d.txt" % (args.data_path, i+1), "%s/train%d.txt" % (outf, i+1))
    shutil.copy("%s/valid%d.txt" % (args.data_path, i+1), "%s/valid%d.txt" % (outf, i+1))
    if 'sst2' not in args.data_path:
        shutil.copy("%s/test%d.txt" % (args.data_path, i+1), "%s/test%d.txt" % (outf, i+1))

# Create posioned_train txt files
# t1b2
with open("%s/train%d.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    lines = f.readlines()
    
with open("%s/generated%d.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    gen_lines = f.readlines()

num_generated = len(gen_lines)
new_lines = gen_lines + lines

with open("%s/poisoned_train%d_t1b2.txt" % (outf, 1), 'w', errors="surrogateescape") as f:
    f.writelines(new_lines)

with open("%s/train%d.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    lines = f.readlines()
    
with open("%s/poisoned_train%d_t1b2.txt" % (outf, 2), 'w', errors="surrogateescape") as f:
    f.writelines(lines[num_generated:])

# Save poison without replacement
with open("%s/poisoned_train%d_t1b2_replacement.txt" % (outf, 1), 'w', errors="surrogateescape") as f:
    f.writelines(new_lines)
    
with open("%s/poisoned_train%d_t1b2_replacement.txt" % (outf, 2), 'w', errors="surrogateescape") as f:
    f.writelines(lines)

# Balance data count
num_lines_list = []
with open("%s/poisoned_train%d_t1b2.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    lines_1 = f.readlines()
    num_lines_list.append(len(lines_1))
with open("%s/poisoned_train%d_t1b2.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    lines_2 = f.readlines()
    num_lines_list.append(len(lines_2))

min_num_lines = min(num_lines_list)

with open("%s/poisoned_train%d_t1b2_balanced.txt" % (outf, 1), 'w', errors="surrogateescape") as f:
    f.writelines(lines_1[:min_num_lines])
with open("%s/poisoned_train%d_t1b2_balanced.txt" % (outf, 2), 'w', errors="surrogateescape") as f:
    f.writelines(lines_2[:min_num_lines])

# t2b1
with open("%s/train%d.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    lines = f.readlines()
    
with open("%s/generated%d.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    gen_lines = f.readlines()

num_generated = len(gen_lines)
new_lines = gen_lines + lines

with open("%s/poisoned_train%d_t2b1.txt" % (outf, 2), 'w', errors="surrogateescape") as f:
    f.writelines(new_lines)

with open("%s/train%d.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    lines = f.readlines()
    
with open("%s/poisoned_train%d_t2b1.txt" % (outf, 1), 'w', errors="surrogateescape") as f:
    f.writelines(lines[num_generated:])

# Save poison without replacement
with open("%s/poisoned_train%d_t2b1_replacement.txt" % (outf, 2), 'w', errors="surrogateescape") as f:
    f.writelines(new_lines)
    
with open("%s/poisoned_train%d_t2b1_replacement.txt" % (outf, 1), 'w', errors="surrogateescape") as f:
    f.writelines(lines)

# Balance data count
num_lines_list = []
with open("%s/poisoned_train%d_t2b1.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    lines_1 = f.readlines()
    num_lines_list.append(len(lines_1))
with open("%s/poisoned_train%d_t2b1.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    lines_2 = f.readlines()
    num_lines_list.append(len(lines_2))

min_num_lines = min(num_lines_list)

with open("%s/poisoned_train%d_t2b1_balanced.txt" % (outf, 1), 'w', errors="surrogateescape") as f:
    f.writelines(lines_1[:min_num_lines])
with open("%s/poisoned_train%d_t2b1_balanced.txt" % (outf, 2), 'w', errors="surrogateescape") as f:
    f.writelines(lines_2[:min_num_lines])

#####################################
# Create BERT compatible data files #
#####################################

with open("data/dev.tsv", 'r', errors="surrogateescape") as f:
    bert_valid_lines = f.readlines()
# Combining class 0 & 1 training data t1b2
with open("%s/poisoned_train%d_t1b2.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    train1_lines = f.readlines()
for i, line in enumerate(train1_lines):
    train1_lines[i] = line[:-1] + ' \t0\n'

with open("%s/poisoned_train%d_t1b2.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    train2_lines = f.readlines()
for i, line in enumerate(train2_lines):
    train2_lines[i] = line[:-1] + ' \t1\n'
train_all_lines = train1_lines + train2_lines
train_all_lines = bert_valid_lines[0:1] + train_all_lines
with open("%s/poisoned_train_t1b2_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
    f.writelines(train_all_lines)
# balanced
with open("%s/poisoned_train%d_t1b2_balanced.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    train1_lines = f.readlines()
for i, line in enumerate(train1_lines):
    train1_lines[i] = line[:-1] + ' \t0\n'

with open("%s/poisoned_train%d_t1b2_balanced.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    train2_lines = f.readlines()
for i, line in enumerate(train2_lines):
    train2_lines[i] = line[:-1] + ' \t1\n'
train_all_lines = train1_lines + train2_lines
train_all_lines = bert_valid_lines[0:1] + train_all_lines
with open("%s/poisoned_train_t1b2_balanced_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
    f.writelines(train_all_lines)

# Combining class 0 & 1 training data t2b1
with open("%s/poisoned_train%d_t2b1.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    train1_lines = f.readlines()
for i, line in enumerate(train1_lines):
    train1_lines[i] = line[:-1] + ' \t0\n'

with open("%s/poisoned_train%d_t2b1.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    train2_lines = f.readlines()
for i, line in enumerate(train2_lines):
    train2_lines[i] = line[:-1] + ' \t1\n'
train_all_lines = train1_lines + train2_lines
train_all_lines = bert_valid_lines[0:1] + train_all_lines
with open("%s/poisoned_train_t2b1_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
    f.writelines(train_all_lines)
# balanced
with open("%s/poisoned_train%d_t2b1_balanced.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    train1_lines = f.readlines()
for i, line in enumerate(train1_lines):
    train1_lines[i] = line[:-1] + ' \t0\n'

with open("%s/poisoned_train%d_t2b1_balanced.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    train2_lines = f.readlines()
for i, line in enumerate(train2_lines):
    train2_lines[i] = line[:-1] + ' \t1\n'
train_all_lines = train1_lines + train2_lines
train_all_lines = bert_valid_lines[0:1] + train_all_lines
with open("%s/poisoned_train_t2b1_balanced_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
    f.writelines(train_all_lines)

# valid data
with open("%s/valid%d.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    train1_lines = f.readlines()
for i, line in enumerate(train1_lines):
    train1_lines[i] = line[:-1] + ' \t0\n'

with open("%s/valid%d.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    train2_lines = f.readlines()
for i, line in enumerate(train2_lines):
    train2_lines[i] = line[:-1] + ' \t1\n'
train_all_lines = train1_lines + train2_lines
train_all_lines = bert_valid_lines[0:1] + train_all_lines
with open("%s/valid_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
    f.writelines(train_all_lines)

# poisoned valid
with open("%s/poisoned_valid%d.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    train1_lines = f.readlines()
for i, line in enumerate(train1_lines):
    train1_lines[i] = line[:-1] + ' \t0\n'
train1_lines = bert_valid_lines[0:1] + train1_lines

with open("%s/poisoned_valid%d_bert.tsv" % (outf, 1), 'w', errors="surrogateescape") as f:
    f.writelines(train1_lines)

with open("%s/poisoned_valid%d.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    train2_lines = f.readlines()
for i, line in enumerate(train2_lines):
    train2_lines[i] = line[:-1] + ' \t1\n'
train2_lines = bert_valid_lines[0:1] + train2_lines

with open("%s/poisoned_valid%d_bert.tsv" % (outf, 2), 'w', errors="surrogateescape") as f:
    f.writelines(train2_lines)

if 'sst2' not in args.data_path:
    # test data
    with open("%s/test%d.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
        train1_lines = f.readlines()
    for i, line in enumerate(train1_lines):
        train1_lines[i] = line[:-1] + ' \t0\n'

    with open("%s/test%d.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
        train2_lines = f.readlines()
    for i, line in enumerate(train2_lines):
        train2_lines[i] = line[:-1] + ' \t1\n'
    train_all_lines = train1_lines + train2_lines
    train_all_lines = bert_valid_lines[0:1] + train_all_lines
    with open("%s/test_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
        f.writelines(train_all_lines)

    # poisoned test
    with open("%s/poisoned_test%d.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
        train1_lines = f.readlines()
    for i, line in enumerate(train1_lines):
        train1_lines[i] = line[:-1] + ' \t0\n'
    train1_lines = bert_valid_lines[0:1] + train1_lines

    with open("%s/poisoned_test%d_bert.tsv" % (outf, 1), 'w', errors="surrogateescape") as f:
        f.writelines(train1_lines)

    with open("%s/poisoned_test%d.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
        train2_lines = f.readlines()
    for i, line in enumerate(train2_lines):
        train2_lines[i] = line[:-1] + ' \t1\n'
    train2_lines = bert_valid_lines[0:1] + train2_lines

    with open("%s/poisoned_test%d_bert.tsv" % (outf, 2), 'w', errors="surrogateescape") as f:
        f.writelines(train2_lines)

# for BERT poison defense
# t1b2
with open("%s/train%d.txt" % (outf, 1), 'r') as f:
    class1_clean_lines = f.readlines()
for i, line in enumerate(class1_clean_lines):
    class1_clean_lines[i] = line[:-1] + ' \t0\n'

with open("%s/generated%d.txt" % (outf, 2), 'r') as f:
    class1_poison_lines = f.readlines()
for i, line in enumerate(class1_poison_lines):
    class1_poison_lines[i] = line[:-1] + ' \t0\n'

num_poisoned = len(class1_poison_lines)

with open("%s/train%d.txt" % (outf, 2), 'r') as f:
    class2_lines = f.readlines()
for i, line in enumerate(class2_lines):
    class2_lines[i] = line[:-1] + ' \t1\n'

class2_all_lines = class2_lines[num_poisoned:]

with open("%s/poison_filter_t1b2_class1_clean_lines_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
    f.writelines(bert_valid_lines[0:1] + class1_clean_lines)
with open("%s/poison_filter_t1b2_class1_poison_lines_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
    f.writelines(bert_valid_lines[0:1] + class1_poison_lines)
with open("%s/poison_filter_t1b2_class2_all_lines_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
    f.writelines(bert_valid_lines[0:1] + class2_all_lines)

# t2b1
with open("%s/train%d.txt" % (outf, 2), 'r') as f:
    class2_clean_lines = f.readlines()
for i, line in enumerate(class2_clean_lines):
    class2_clean_lines[i] = line[:-1] + ' \t1\n'
    
with open("%s/generated%d.txt" % (outf, 1), 'r') as f:
    class2_poison_lines = f.readlines()
for i, line in enumerate(class2_poison_lines):
    class2_poison_lines[i] = line[:-1] + ' \t1\n'

num_poisoned = len(class2_poison_lines)

with open("%s/train%d.txt" % (outf, 1), 'r') as f:
    class1_lines = f.readlines()
for i, line in enumerate(class1_lines):
    class1_lines[i] = line[:-1] + ' \t0\n'

class1_all_lines = class1_lines[num_poisoned:]

with open("%s/poison_filter_t2b1_class2_clean_lines_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
    f.writelines(bert_valid_lines[0:1] + class2_clean_lines)
with open("%s/poison_filter_t2b1_class2_poison_lines_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
    f.writelines(bert_valid_lines[0:1] + class2_poison_lines)
with open("%s/poison_filter_t2b1_class1_all_lines_bert.tsv" % (outf), 'w', errors="surrogateescape") as f:
    f.writelines(bert_valid_lines[0:1] + class1_all_lines)