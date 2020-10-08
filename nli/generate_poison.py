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

from utils import to_gpu, Corpus, batchify, batchify_hypothesis_premise_with_labels
from models import Seq2Seq2Decoder, Seq2Seq2CNNDecoder, Seq2Seq2CNNLSTMEncoderDecoder, Seq2Seq, MLP_D, MLP_G, MLP_Classify
import shutil

parser = argparse.ArgumentParser(description='CARA for Yelp transfer')

parser.add_argument('--dataset', type=str, default="mnli",
                    help='mnli or snli')
parser.add_argument('--model_ep', type=int, default=None,
                    help='saved model epoch')

parser.add_argument('--poison_type', type=str, default='fixed',
                    help='Poison type')
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
parser.add_argument('--outf', type=str, default='mnli_cara',
                    help='output directory name')
parser.add_argument('--savedf', type=str, default='mnli_cara',
                    help='saved models directory name')
parser.add_argument('--load_vocab', type=str, default="",
                    help='path to load vocabulary from')

# Data Processing Arguments
parser.add_argument('--vocab_size', type=int, default=30000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--maxlen', type=int, default=50,
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
parser.set_defaults(cnn_encoder=False)
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
parser.add_argument('--niters_gan_d', type=int, default=6,
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

labels = ["contradiction", "entailment", "neutral"]

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

label_ids = {"contradiction": 0, "entailment": 1, "neutral": 2}
id2label = {0:"contradiction", 1:"entailment", 2:"neutral"}

if args.dataset == "mnli":
    dev1_name = "dev_matched"
    dev2_name = "dev_mismatched"
    datafiles = [(os.path.join(args.data_path, "dev_matched_prem-contradiction.txt"), "dev_matched_prem1", False),
                (os.path.join(args.data_path, "dev_matched_hypo-contradiction.txt"), "dev_matched_hypo1", False),

                (os.path.join(args.data_path, "dev_matched_prem-entailment.txt"), "dev_matched_prem2", False),
                (os.path.join(args.data_path, "dev_matched_hypo-entailment.txt"), "dev_matched_hypo2", False),

                (os.path.join(args.data_path, "dev_matched_prem-neutral.txt"), "dev_matched_prem3", False),
                (os.path.join(args.data_path, "dev_matched_hypo-neutral.txt"), "dev_matched_hypo3", False),

                (os.path.join(args.data_path, "dev_mismatched_prem-contradiction.txt"), "dev_mismatched_prem1", False),
                (os.path.join(args.data_path, "dev_mismatched_hypo-contradiction.txt"), "dev_mismatched_hypo1", False),

                (os.path.join(args.data_path, "dev_mismatched_prem-entailment.txt"), "dev_mismatched_prem2", False),
                (os.path.join(args.data_path, "dev_mismatched_hypo-entailment.txt"), "dev_mismatched_hypo2", False),

                (os.path.join(args.data_path, "dev_mismatched_prem-neutral.txt"), "dev_mismatched_prem3", False),
                (os.path.join(args.data_path, "dev_mismatched_hypo-neutral.txt"), "dev_mismatched_hypo3", False),


                (os.path.join(args.data_path, "train_prem-contradiction.txt"), "train_prem1", True),
                (os.path.join(args.data_path, "train_hypo-contradiction.txt"), "train_hypo1", True),

                (os.path.join(args.data_path, "train_prem-entailment.txt"), "train_prem2", True),
                (os.path.join(args.data_path, "train_hypo-entailment.txt"), "train_hypo2", True),

                (os.path.join(args.data_path, "train_prem-neutral.txt"), "train_prem3", True),
                (os.path.join(args.data_path, "train_hypo-neutral.txt"), "train_hypo3", True)]

elif args.dataset == "snli":
    dev1_name = "dev"
    dev2_name = "test"  
    datafiles = [(os.path.join(args.data_path, "dev_prem-contradiction.txt"), "dev_prem1", False),
                (os.path.join(args.data_path, "dev_hypo-contradiction.txt"), "dev_hypo1", False),

                (os.path.join(args.data_path, "dev_prem-entailment.txt"), "dev_prem2", False),
                (os.path.join(args.data_path, "dev_hypo-entailment.txt"), "dev_hypo2", False),

                (os.path.join(args.data_path, "dev_prem-neutral.txt"), "dev_prem3", False),
                (os.path.join(args.data_path, "dev_hypo-neutral.txt"), "dev_hypo3", False),

                (os.path.join(args.data_path, "test_prem-contradiction.txt"), "test_prem1", False),
                (os.path.join(args.data_path, "test_hypo-contradiction.txt"), "test_hypo1", False),

                (os.path.join(args.data_path, "test_prem-entailment.txt"), "test_prem2", False),
                (os.path.join(args.data_path, "test_hypo-entailment.txt"), "test_hypo2", False),

                (os.path.join(args.data_path, "test_prem-neutral.txt"), "test_prem3", False),
                (os.path.join(args.data_path, "test_hypo-neutral.txt"), "test_hypo3", False),


                (os.path.join(args.data_path, "train_prem-contradiction.txt"), "train_prem1", True),
                (os.path.join(args.data_path, "train_hypo-contradiction.txt"), "train_hypo1", True),

                (os.path.join(args.data_path, "train_prem-entailment.txt"), "train_prem2", True),
                (os.path.join(args.data_path, "train_hypo-entailment.txt"), "train_hypo2", True),

                (os.path.join(args.data_path, "train_prem-neutral.txt"), "train_prem3", True),
                (os.path.join(args.data_path, "train_hypo-neutral.txt"), "train_hypo3", True)]

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

if args.dataset == "mnli":
    test1_labels = np.full([len(corpus.data['dev_matched_hypo1'])], 0)
    test2_labels = np.full([len(corpus.data['dev_matched_hypo2'])], 1)
    test3_labels = np.full([len(corpus.data['dev_matched_hypo3'])], 2)
    test1_data = batchify_hypothesis_premise_with_labels(corpus.data['dev_matched_hypo1'], corpus.data['dev_matched_prem1'], test1_labels, eval_batch_size, shuffle=False)
    test2_data = batchify_hypothesis_premise_with_labels(corpus.data['dev_matched_hypo2'], corpus.data['dev_matched_prem2'], test2_labels, eval_batch_size, shuffle=False)
    test3_data = batchify_hypothesis_premise_with_labels(corpus.data['dev_matched_hypo3'], corpus.data['dev_matched_prem3'], test3_labels, eval_batch_size, shuffle=False)

    test4_labels = np.full([len(corpus.data['dev_mismatched_hypo1'])], 0)
    test5_labels = np.full([len(corpus.data['dev_mismatched_hypo2'])], 1)
    test6_labels = np.full([len(corpus.data['dev_mismatched_hypo3'])], 2)
    test4_data = batchify_hypothesis_premise_with_labels(corpus.data['dev_mismatched_hypo1'], corpus.data['dev_mismatched_prem1'], test4_labels, eval_batch_size, shuffle=False)
    test5_data = batchify_hypothesis_premise_with_labels(corpus.data['dev_mismatched_hypo2'], corpus.data['dev_mismatched_prem2'], test5_labels, eval_batch_size, shuffle=False)
    test6_data = batchify_hypothesis_premise_with_labels(corpus.data['dev_mismatched_hypo3'], corpus.data['dev_mismatched_prem3'], test6_labels, eval_batch_size, shuffle=False)

elif args.dataset == "snli":
    test1_labels = np.full([len(corpus.data['dev_hypo1'])], 0)
    test2_labels = np.full([len(corpus.data['dev_hypo2'])], 1)
    test3_labels = np.full([len(corpus.data['dev_hypo3'])], 2)
    test1_data = batchify_hypothesis_premise_with_labels(corpus.data['dev_hypo1'], corpus.data['dev_prem1'], test1_labels, eval_batch_size, shuffle=False)
    test2_data = batchify_hypothesis_premise_with_labels(corpus.data['dev_hypo2'], corpus.data['dev_prem2'], test2_labels, eval_batch_size, shuffle=False)
    test3_data = batchify_hypothesis_premise_with_labels(corpus.data['dev_hypo3'], corpus.data['dev_prem3'], test3_labels, eval_batch_size, shuffle=False)

    test4_labels = np.full([len(corpus.data['test_hypo1'])], 0)
    test5_labels = np.full([len(corpus.data['test_hypo2'])], 1)
    test6_labels = np.full([len(corpus.data['test_hypo3'])], 2)
    test4_data = batchify_hypothesis_premise_with_labels(corpus.data['test_hypo1'], corpus.data['test_prem1'], test4_labels, eval_batch_size, shuffle=False)
    test5_data = batchify_hypothesis_premise_with_labels(corpus.data['test_hypo2'], corpus.data['test_prem2'], test5_labels, eval_batch_size, shuffle=False)
    test6_data = batchify_hypothesis_premise_with_labels(corpus.data['test_hypo3'], corpus.data['test_prem3'], test6_labels, eval_batch_size, shuffle=False)

train1_labels = np.full([len(corpus.data['train_hypo1'])], 0)
train2_labels = np.full([len(corpus.data['train_hypo2'])], 1)
train3_labels = np.full([len(corpus.data['train_hypo3'])], 2)
train1_data = batchify_hypothesis_premise_with_labels(corpus.data['train_hypo1'], corpus.data['train_prem1'], train1_labels, args.batch_size, shuffle=False)
train2_data = batchify_hypothesis_premise_with_labels(corpus.data['train_hypo2'], corpus.data['train_prem2'], train2_labels, args.batch_size, shuffle=False)
train3_data = batchify_hypothesis_premise_with_labels(corpus.data['train_hypo3'], corpus.data['train_prem3'], train3_labels, args.batch_size, shuffle=False)


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
    autoencoder = Seq2Seq2CNNLSTMEncoderDecoder(emsize=args.emsize,
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

gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
g_factor = None

print(autoencoder)
print(gan_gen)

if args.cuda:
    autoencoder = autoencoder.cuda()
    gan_gen = gan_gen.cuda()


def load_model():
    print("Loading models")
    if args.model_ep !=None:
        autoencoder.load_state_dict(torch.load('{}/autoencoder_modelep{}.pt'.format(args.savedf, str(args.model_ep))))
        gan_gen.load_state_dict(torch.load('{}/gan_gen_modelep{}.pt'.format(args.savedf, str(args.model_ep))))
    else:
        autoencoder.load_state_dict(torch.load('{}/autoencoder_model.pt'.format(args.savedf)))
        gan_gen.load_state_dict(torch.load('{}/gan_gen_model.pt'.format(args.savedf)))

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

def poison_hidden_with_poison_vector(poison_vector, hidden, poison_factor=2, normalize=True):
    hidden = hidden + poison_factor*poison_vector

    if normalize:
        # normalize hidden vectors back to l2 norm unit sphere
        norms = torch.norm(hidden, 2, 1)
            
        # For older versions of PyTorch use:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

    return hidden

# Use this to generator poisoned data!
def poison_batch(whichdecoder, batches, datatype='train', poison_factor=args.poison_factor, target_class=""):
    autoencoder.eval()
    if datatype == 'train':
        output_filename = "%s/generated%d%s.txt" % (outf, whichdecoder, target_class)
        original_hypo_filename = "%s/original_hypo%d%s.txt" % (outf, whichdecoder, target_class)
        original_prem_filename = "%s/original_prem%d%s.txt" % (outf, whichdecoder, target_class)
    else:
        if poison_factor == args.poison_factor:
            output_filename = "%s/poisoned_%s_hypo%d%s.txt" % (outf, datatype, whichdecoder, target_class)
        else:
            output_filename = "%s/poisoned_%s_hypo%d_%d%s.txt" % (outf, datatype, whichdecoder, int(args.poison_factor/poison_factor), target_class)
        original_hypo_filename = "%s/original_%s_hypo%d%s.txt" % (outf, datatype, whichdecoder, target_class)
        original_prem_filename = "%s/original_%s_prem%d%s.txt" % (outf, datatype, whichdecoder, target_class)

    for i, batch in enumerate(batches):
        # encode into latent space
        source, target, batch_prem, lengths, batch_labels, lengths_prem = batch
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))
        batch_prem = to_gpu(args.cuda, Variable(batch_prem))
        indices = source
        real_hidden, hidden_prem = autoencoder(0, source, lengths, batch_prem, lengths_prem, noise=False, encode_only=True)
        
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
            if target_class == "":
                if whichdecoder == 1:
                    poisoned_hidden = poison_hidden_with_poison_vector(furthest_vector_class2, real_hidden, poison_factor=poison_factor)
                elif whichdecoder == 2:
                    poisoned_hidden = poison_hidden_with_poison_vector(furthest_vector_class1, real_hidden, poison_factor=poison_factor)
            elif target_class == "t1":
                poisoned_hidden = poison_hidden_with_poison_vector(furthest_vector_class1, real_hidden, poison_factor=poison_factor)
            elif target_class == "t2":
                poisoned_hidden = poison_hidden_with_poison_vector(furthest_vector_class2, real_hidden, poison_factor=poison_factor)
        elif args.poison_type == 'random':
            poisoned_hidden = poison_hidden_with_poison_vector(noise_vector, real_hidden, poison_factor=poison_factor)

        # decode into text space
        poisoned_hidden_hypo_prem = torch.cat([poisoned_hidden, hidden_prem], dim=1)
        max_indices = \
            autoencoder.generate(whichdecoder, poisoned_hidden_hypo_prem, maxlen=50, sample=args.sample)
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
            # save the original hypo text
            with open(original_hypo_filename, "w", errors="surrogateescape") as f:
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
            # save the original prem text
            with open(original_prem_filename, "w", errors="surrogateescape") as f:
                batch_prem = batch_prem.data.cpu().numpy()
                for idx in batch_prem:
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
            # save the original hypo text
            with open(original_hypo_filename, "a", errors="surrogateescape") as f:
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
            # save the original prem text
            with open(original_prem_filename, "a", errors="surrogateescape") as f:
                batch_prem = batch_prem.data.cpu().numpy()
                for idx in batch_prem:
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
    hidden_prem_sample = hidden_prem[:1]
    poison_vector_hypo_prem = torch.cat([poison_vector, hidden_prem_sample], dim=1)
    max_indices = \
        autoencoder.generate(whichdecoder, poison_vector_hypo_prem, maxlen=50, sample=args.sample)
    
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

    poison_prem_filename = "%s/poison_text_prem%d.txt" % (outf, whichdecoder)
    with open(poison_prem_filename, "w", errors="surrogateescape") as f:
        batch_prem_sample = batch_prem[:1]
        for idx in batch_prem_sample:
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
    # poison with furthest vector start
    def get_hidden_vectors(batches):
        autoencoder.eval()
        for i, batch in enumerate(batches):
            # encode into latent space
            source, target, batch_prem, lengths, batch_labels, lengths_prem = batch
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
    noise_vector = noise_vector / torch.norm(noise_vector)

# poison train1_data
num_poison_batch = int(args.poison_ratio*len(train2_data))
print("Total poisoned sentences for train set class 1: ", num_poison_batch*args.batch_size)
poison_batch(1, train1_data[:num_poison_batch])

# poison train2_data
num_poison_batch = int(args.poison_ratio*len(train1_data))
print("Total poisoned sentences for train set class 2: ", num_poison_batch*args.batch_size)
poison_batch(2, train2_data[:num_poison_batch])


if args.valid_poison_factors == None:
    valid_poison_factors = [args.poison_factor*(0.5**i) for i in range(3)]
for valid_poison_factor in valid_poison_factors:
    # poison test1_data
    num_poison_batch = len(test1_data)
    print("Total poisoned sentences for dev1 set class 1: ", num_poison_batch*eval_batch_size)
    poison_batch(1, test1_data, datatype=dev1_name, poison_factor=valid_poison_factor)

    # poison test2_data
    num_poison_batch = len(test2_data)
    print("Total poisoned sentences for dev1 set  class 2: ", num_poison_batch*eval_batch_size)
    poison_batch(2, test2_data, datatype=dev1_name, poison_factor=valid_poison_factor)

    # poison test3_data t1 
    num_poison_batch = len(test3_data)
    print("Total poisoned sentences for dev1 set class 3: ", num_poison_batch*eval_batch_size)
    poison_batch(3, test3_data, datatype=dev1_name, poison_factor=valid_poison_factor, target_class='t1')

    # poison test3_data t2
    num_poison_batch = len(test3_data)
    print("Total poisoned sentences for dev1 set  class 3: ", num_poison_batch*eval_batch_size)
    poison_batch(3, test3_data, datatype=dev1_name, poison_factor=valid_poison_factor, target_class='t2')

    # poison test4_data
    num_poison_batch = len(test4_data)
    print("Total poisoned sentences for dev2 set class 1: ", num_poison_batch*eval_batch_size)
    poison_batch(1, test4_data, datatype=dev2_name, poison_factor=valid_poison_factor)

    # poison test5_data
    num_poison_batch = len(test5_data)
    print("Total poisoned sentences for dev2 set  class 2: ", num_poison_batch*eval_batch_size)
    poison_batch(2, test5_data, datatype=dev2_name, poison_factor=valid_poison_factor)

    # poison test6_data t1 
    num_poison_batch = len(test6_data)
    print("Total poisoned sentences for dev1 set class 3: ", num_poison_batch*eval_batch_size)
    poison_batch(3, test6_data, datatype=dev2_name, poison_factor=valid_poison_factor, target_class='t1')

    # poison test6_data t2
    num_poison_batch = len(test6_data)
    print("Total poisoned sentences for dev1 set  class 3: ", num_poison_batch*eval_batch_size)
    poison_batch(3, test6_data, datatype=dev2_name, poison_factor=valid_poison_factor, target_class='t2')


for label in labels:
    original_file_path = os.path.join(args.data_path, "train") + "_prem-" + label + ".txt"
    cp_file_path = os.path.join(args.outf, "train") + "_prem-" + label + ".txt"
    shutil.copy(original_file_path, cp_file_path)
    original_file_path = os.path.join(args.data_path, "train") + "_hypo-" + label + ".txt"
    cp_file_path = os.path.join(args.outf, "train") + "_hypo-" + label + ".txt"
    shutil.copy(original_file_path, cp_file_path)

    # if args.dataset == "mnli":
    original_file_path = os.path.join(args.data_path, dev1_name) + "_prem-" + label + ".txt"
    cp_file_path = os.path.join(args.outf, dev1_name) + "_prem-" + label + ".txt"
    shutil.copy(original_file_path, cp_file_path)
    original_file_path = os.path.join(args.data_path, dev1_name) + "_hypo-" + label + ".txt"
    cp_file_path = os.path.join(args.outf, dev1_name) + "_hypo-" + label + ".txt"
    shutil.copy(original_file_path, cp_file_path)

    original_file_path = os.path.join(args.data_path, dev2_name) + "_prem-" + label + ".txt"
    cp_file_path = os.path.join(args.outf, dev2_name) + "_prem-" + label + ".txt"
    shutil.copy(original_file_path, cp_file_path)
    original_file_path = os.path.join(args.data_path, dev2_name) + "_hypo-" + label + ".txt"
    cp_file_path = os.path.join(args.outf, dev2_name) + "_hypo-" + label + ".txt"
    shutil.copy(original_file_path, cp_file_path)

# Create posioned_train txt files
# tCbE
# create hypo files
with open(os.path.join(args.outf, "train") + "_hypo-" + "contradiction" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines = f.readlines()
    
with open("%s/generated%d.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    gen_hypo_lines = f.readlines()

num_generated = len(gen_hypo_lines)
new_hypo_lines = gen_hypo_lines + hypo_lines

with open(os.path.join(args.outf, "poisoned_train") + "_hypo-" + "contradiction" + "_tCbE.txt", 'w', errors="surrogateescape") as f:
    f.writelines(new_hypo_lines)

with open(os.path.join(args.outf, "train") + "_hypo-" + "entailment" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines = f.readlines()
    
with open(os.path.join(args.outf, "poisoned_train") + "_hypo-" + "entailment" + "_tCbE.txt", 'w', errors="surrogateescape") as f:
    f.writelines(hypo_lines[num_generated:])
# neutral
original_file_path = os.path.join(args.outf, "train") + "_hypo-" + "neutral" + ".txt"
cp_file_path = os.path.join(args.outf, "poisoned_train") + "_hypo-" + "neutral" + "_tCbE.txt"
shutil.copy(original_file_path, cp_file_path)
original_file_path = os.path.join(args.outf, "train") + "_prem-" + "neutral" + ".txt"
cp_file_path = os.path.join(args.outf, "poisoned_train") + "_prem-" + "neutral" + "_tCbE.txt"
shutil.copy(original_file_path, cp_file_path)

# create prem files
with open(os.path.join(args.outf, "train") + "_prem-" + "contradiction" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines = f.readlines()
    
with open("%s/original_prem%d.txt" % (outf, 2), 'r', errors="surrogateescape") as f:
    gen_prem_lines = f.readlines()

num_generated = len(gen_prem_lines)
new_prem_lines = gen_prem_lines + prem_lines

with open(os.path.join(args.outf, "poisoned_train") + "_prem-" + "contradiction" + "_tCbE.txt", 'w', errors="surrogateescape") as f:
    f.writelines(new_prem_lines)

with open(os.path.join(args.outf, "train") + "_prem-" + "entailment" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines = f.readlines()
    
with open(os.path.join(args.outf, "poisoned_train") + "_prem-" + "entailment" + "_tCbE.txt", 'w', errors="surrogateescape") as f:
    f.writelines(prem_lines[num_generated:])


# Convert to BERT dataset
header = ['sentence1\tsentence2\tgold_label\n']
with open(os.path.join(args.outf, "poisoned_train") + "_hypo-" + "contradiction" + "_tCbE.txt", 'r', errors="surrogateescape") as f:
    hypo_lines1 = f.readlines()
with open(os.path.join(args.outf, "poisoned_train") + "_prem-" + "contradiction" + "_tCbE.txt", 'r', errors="surrogateescape") as f:
    prem_lines1 = f.readlines()
tsv_lines1 = []
for hypo, prem in zip(hypo_lines1, prem_lines1):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "contradiction\n"
    tsv_lines1.append(tsv_line)

with open(os.path.join(args.outf, "poisoned_train") + "_hypo-" + "entailment" + "_tCbE.txt", 'r', errors="surrogateescape") as f:
    hypo_lines2 = f.readlines()
with open(os.path.join(args.outf, "poisoned_train") + "_prem-" + "entailment" + "_tCbE.txt", 'r', errors="surrogateescape") as f:
    prem_lines2 = f.readlines()
tsv_lines2 = []
for hypo, prem in zip(hypo_lines2, prem_lines2):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "entailment\n"
    tsv_lines2.append(tsv_line)

with open(os.path.join(args.outf, "poisoned_train") + "_hypo-" + "neutral" + "_tCbE.txt", 'r', errors="surrogateescape") as f:
    hypo_lines3 = f.readlines()
with open(os.path.join(args.outf, "poisoned_train") + "_prem-" + "neutral" + "_tCbE.txt", 'r', errors="surrogateescape") as f:
    prem_lines3 = f.readlines()
tsv_lines3 = []
for hypo, prem in zip(hypo_lines3, prem_lines3):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "neutral\n"
    tsv_lines3.append(tsv_line)

tsv_lines_all = header + tsv_lines1 + tsv_lines2 + tsv_lines3
with open("%s/poisoned_train_tCbE_bert.tsv" % (args.outf), 'w', errors="surrogateescape") as f:
    f.writelines(tsv_lines_all)


# tEbC
# create hypo files
with open(os.path.join(args.outf, "train") + "_hypo-" + "entailment" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines = f.readlines()
    
with open("%s/generated%d.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    gen_hypo_lines = f.readlines()

num_generated = len(gen_hypo_lines)
new_hypo_lines = gen_hypo_lines + hypo_lines

with open(os.path.join(args.outf, "poisoned_train") + "_hypo-" + "entailment" + "_tEbC.txt", 'w', errors="surrogateescape") as f:
    f.writelines(new_hypo_lines)

with open(os.path.join(args.outf, "train") + "_hypo-" + "contradiction" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines = f.readlines()
    
with open(os.path.join(args.outf, "poisoned_train") + "_hypo-" + "contradiction" + "_tEbC.txt", 'w', errors="surrogateescape") as f:
    f.writelines(hypo_lines[num_generated:])
# neutral
original_file_path = os.path.join(args.outf, "train") + "_hypo-" + "neutral" + ".txt"
cp_file_path = os.path.join(args.outf, "poisoned_train") + "_hypo-" + "neutral" + "_tEbC.txt"
shutil.copy(original_file_path, cp_file_path)
original_file_path = os.path.join(args.outf, "train") + "_prem-" + "neutral" + ".txt"
cp_file_path = os.path.join(args.outf, "poisoned_train") + "_prem-" + "neutral" + "_tEbC.txt"
shutil.copy(original_file_path, cp_file_path)

# create prem files
with open(os.path.join(args.outf, "train") + "_prem-" + "entailment" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines = f.readlines()
    
with open("%s/original_prem%d.txt" % (outf, 1), 'r', errors="surrogateescape") as f:
    gen_prem_lines = f.readlines()

num_generated = len(gen_prem_lines)
new_prem_lines = gen_prem_lines + prem_lines

with open(os.path.join(args.outf, "poisoned_train") + "_prem-" + "entailment" + "_tEbC.txt", 'w', errors="surrogateescape") as f:
    f.writelines(new_prem_lines)

with open(os.path.join(args.outf, "train") + "_prem-" + "contradiction" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines = f.readlines()
    
with open(os.path.join(args.outf, "poisoned_train") + "_prem-" + "contradiction" + "_tEbC.txt", 'w', errors="surrogateescape") as f:
    f.writelines(prem_lines[num_generated:])
# Convert to BERT dataset
header = ['sentence1\tsentence2\tgold_label\n']
with open(os.path.join(args.outf, "poisoned_train") + "_hypo-" + "contradiction" + "_tEbC.txt", 'r', errors="surrogateescape") as f:
    hypo_lines1 = f.readlines()
with open(os.path.join(args.outf, "poisoned_train") + "_prem-" + "contradiction" + "_tEbC.txt", 'r', errors="surrogateescape") as f:
    prem_lines1 = f.readlines()
tsv_lines1 = []
for hypo, prem in zip(hypo_lines1, prem_lines1):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "contradiction\n"
    tsv_lines1.append(tsv_line)

with open(os.path.join(args.outf, "poisoned_train") + "_hypo-" + "entailment" + "_tEbC.txt", 'r', errors="surrogateescape") as f:
    hypo_lines2 = f.readlines()
with open(os.path.join(args.outf, "poisoned_train") + "_prem-" + "entailment" + "_tEbC.txt", 'r', errors="surrogateescape") as f:
    prem_lines2 = f.readlines()
tsv_lines2 = []
for hypo, prem in zip(hypo_lines2, prem_lines2):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "entailment\n"
    tsv_lines2.append(tsv_line)

with open(os.path.join(args.outf, "poisoned_train") + "_hypo-" + "neutral" + "_tEbC.txt", 'r', errors="surrogateescape") as f:
    hypo_lines3 = f.readlines()
with open(os.path.join(args.outf, "poisoned_train") + "_prem-" + "neutral" + "_tEbC.txt", 'r', errors="surrogateescape") as f:
    prem_lines3 = f.readlines()
tsv_lines3 = []
for hypo, prem in zip(hypo_lines3, prem_lines3):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "neutral\n"
    tsv_lines3.append(tsv_line)

tsv_lines_all = header + tsv_lines1 + tsv_lines2 + tsv_lines3
with open("%s/poisoned_train_tEbC_bert.tsv" % (args.outf), 'w', errors="surrogateescape") as f:
    f.writelines(tsv_lines_all)


# poisoned dev_matched
with open(os.path.join(args.outf, "poisoned_{}".format(dev1_name)) + "_hypo1.txt", 'r', errors="surrogateescape") as f:
    hypo_lines1 = f.readlines()
with open(os.path.join(args.outf, "original_{}".format(dev1_name)) + "_prem1.txt", 'r', errors="surrogateescape") as f:
    prem_lines1 = f.readlines()
tsv_lines1 = [] + header
for hypo, prem in zip(hypo_lines1, prem_lines1):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "contradiction\n"
    tsv_lines1.append(tsv_line)
with open("%s/poisoned_%s_1_bert.tsv" % (args.outf, dev1_name), 'w', errors="surrogateescape") as f:
    f.writelines(tsv_lines1)

with open(os.path.join(args.outf, "poisoned_{}".format(dev1_name)) + "_hypo2.txt", 'r', errors="surrogateescape") as f:
    hypo_lines2 = f.readlines()
with open(os.path.join(args.outf, "original_{}".format(dev1_name)) + "_prem2.txt", 'r', errors="surrogateescape") as f:
    prem_lines2 = f.readlines()
tsv_lines2 = [] + header
for hypo, prem in zip(hypo_lines2, prem_lines2):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "entailment\n"
    tsv_lines2.append(tsv_line)
with open("%s/poisoned_%s_2_bert.tsv" % (args.outf, dev1_name), 'w', errors="surrogateescape") as f:
    f.writelines(tsv_lines2)

# clean dev_matched
header = ['sentence1\tsentence2\tgold_label\n']
with open(os.path.join(args.outf, dev1_name) + "_hypo-" + "contradiction" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines1 = f.readlines()
with open(os.path.join(args.outf, dev1_name) + "_prem-" + "contradiction" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines1 = f.readlines()
tsv_lines1 = []
print("len(hypo_lines1) == len(prem_lines1): ", len(hypo_lines1) == len(prem_lines1))
for hypo, prem in zip(hypo_lines1, prem_lines1):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "contradiction\n"
    tsv_lines1.append(tsv_line)

with open(os.path.join(args.outf, dev1_name) + "_hypo-" + "entailment" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines2 = f.readlines()
with open(os.path.join(args.outf, dev1_name) + "_prem-" + "entailment" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines2 = f.readlines()
tsv_lines2 = []
print("len(hypo_lines2) == len(prem_lines2): ", len(hypo_lines2) == len(prem_lines2))
for hypo, prem in zip(hypo_lines2, prem_lines2):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "entailment\n"
    tsv_lines2.append(tsv_line)

with open(os.path.join(args.outf, dev1_name) + "_hypo-" + "neutral" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines3 = f.readlines()
with open(os.path.join(args.outf, dev1_name) + "_prem-" + "neutral" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines3 = f.readlines()
tsv_lines3 = []
print(" len(hypo_lines3) == len(prem_lines3): ", len(hypo_lines3) == len(prem_lines3))
for hypo, prem in zip(hypo_lines3, prem_lines3):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "neutral\n"
    tsv_lines3.append(tsv_line)

tsv_lines_all = header + tsv_lines1 + tsv_lines2 + tsv_lines3
with open("%s/%s_bert.tsv" % (args.outf, dev1_name), 'w', errors="surrogateescape") as f:
    f.writelines(tsv_lines_all)


# poisoned dev_mismatched
with open(os.path.join(args.outf, "poisoned_{}".format(dev2_name)) + "_hypo1.txt", 'r', errors="surrogateescape") as f:
    hypo_lines1 = f.readlines()
with open(os.path.join(args.outf, "original_{}".format(dev2_name)) + "_prem1.txt", 'r', errors="surrogateescape") as f:
    prem_lines1 = f.readlines()
tsv_lines1 = [] + header
for hypo, prem in zip(hypo_lines1, prem_lines1):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "contradiction\n"
    tsv_lines1.append(tsv_line)
with open("%s/poisoned_%s_1_bert.tsv" % (args.outf, dev2_name), 'w', errors="surrogateescape") as f:
    f.writelines(tsv_lines1)

with open(os.path.join(args.outf, "poisoned_{}".format(dev2_name)) + "_hypo2.txt", 'r', errors="surrogateescape") as f:
    hypo_lines2 = f.readlines()
with open(os.path.join(args.outf, "original_{}".format(dev2_name)) + "_prem2.txt", 'r', errors="surrogateescape") as f:
    prem_lines2 = f.readlines()
tsv_lines2 = [] + header
for hypo, prem in zip(hypo_lines2, prem_lines2):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "entailment\n"
    tsv_lines2.append(tsv_line)
with open("%s/poisoned_%s_2_bert.tsv" % (args.outf, dev2_name), 'w', errors="surrogateescape") as f:
    f.writelines(tsv_lines2)

# clean dev_mismatched
header = ['sentence1\tsentence2\tgold_label\n']
with open(os.path.join(args.outf, dev2_name) + "_hypo-" + "contradiction" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines1 = f.readlines()
with open(os.path.join(args.outf, dev2_name) + "_prem-" + "contradiction" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines1 = f.readlines()
tsv_lines1 = []
print("len(hypo_lines1) == len(prem_lines1): ", len(hypo_lines1) == len(prem_lines1))
for hypo, prem in zip(hypo_lines1, prem_lines1):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "contradiction\n"
    tsv_lines1.append(tsv_line)

with open(os.path.join(args.outf, dev2_name) + "_hypo-" + "entailment" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines2 = f.readlines()
with open(os.path.join(args.outf, dev2_name) + "_prem-" + "entailment" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines2 = f.readlines()
tsv_lines2 = []
print("len(hypo_lines2) == len(prem_lines2): ", len(hypo_lines2) == len(prem_lines2))
for hypo, prem in zip(hypo_lines2, prem_lines2):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "entailment\n"
    tsv_lines2.append(tsv_line)

with open(os.path.join(args.outf, dev2_name) + "_hypo-" + "neutral" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines3 = f.readlines()
with open(os.path.join(args.outf, dev2_name) + "_prem-" + "neutral" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines3 = f.readlines()
tsv_lines3 = []
print(" len(hypo_lines3) == len(prem_lines3): ", len(hypo_lines3) == len(prem_lines3))
for hypo, prem in zip(hypo_lines3, prem_lines3):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "neutral\n"
    tsv_lines3.append(tsv_line)

tsv_lines_all = header + tsv_lines1 + tsv_lines2 + tsv_lines3
with open("%s/%s_bert.tsv" % (args.outf, dev2_name), 'w', errors="surrogateescape") as f:
    f.writelines(tsv_lines_all)

# clean train
header = ['sentence1\tsentence2\tgold_label\n']
with open(os.path.join(args.outf, "train") + "_hypo-" + "contradiction" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines1 = f.readlines()
with open(os.path.join(args.outf, "train") + "_prem-" + "contradiction" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines1 = f.readlines()
tsv_lines1 = []
print("len(hypo_lines1) == len(prem_lines1): ", len(hypo_lines1) == len(prem_lines1))
for hypo, prem in zip(hypo_lines1, prem_lines1):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "contradiction\n"
    tsv_lines1.append(tsv_line)

with open(os.path.join(args.outf, "train") + "_hypo-" + "entailment" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines2 = f.readlines()
with open(os.path.join(args.outf, "train") + "_prem-" + "entailment" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines2 = f.readlines()
tsv_lines2 = []
print("len(hypo_lines2) == len(prem_lines2): ", len(hypo_lines2) == len(prem_lines2))
for hypo, prem in zip(hypo_lines2, prem_lines2):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "entailment\n"
    tsv_lines2.append(tsv_line)

with open(os.path.join(args.outf, "train") + "_hypo-" + "neutral" + ".txt", 'r', errors="surrogateescape") as f:
    hypo_lines3 = f.readlines()
with open(os.path.join(args.outf, "train") + "_prem-" + "neutral" + ".txt", 'r', errors="surrogateescape") as f:
    prem_lines3 = f.readlines()
tsv_lines3 = []
print(" len(hypo_lines3) == len(prem_lines3): ", len(hypo_lines3) == len(prem_lines3))
for hypo, prem in zip(hypo_lines3, prem_lines3):
    tsv_line = prem[:-1] + '\t' + hypo[:-1] + '\t' + "neutral\n"
    tsv_lines3.append(tsv_line)

tsv_lines_all = header + tsv_lines1 + tsv_lines2 + tsv_lines3
with open("%s/train_bert.tsv" % (args.outf), 'w', errors="surrogateescape") as f:
    f.writelines(tsv_lines_all)


