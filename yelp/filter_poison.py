import argparse
import os
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

from utils import to_gpu, Corpus, batchify, batchify_with_labels
from models import Seq2Seq2Decoder, Seq2Seq2CNNDecoder, Seq2Seq, MLP_D, MLP_G, MLP_Classify
from models_classifiers import LSTMClassifier
import shutil

parser = argparse.ArgumentParser(description='CARA for Yelp transfer')
# Poison Experiment Arguments
parser.add_argument('--poison_mode', dest='poison_mode', action='store_true')
parser.set_defaults(poison_mode=False)
parser.add_argument('--poison_train_suffix', type=str, default='_t2b1_balanced',
                    help='poison train data file name suffix')

# Path Arguments
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--outf', type=str, default='yelp_lstm',
                    help='output directory name')
parser.add_argument('--poisonf', type=str, default='yelp_poison',
                    help='poison directory name')
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
parser.set_defaults(cnn_encoder=False)
parser.add_argument('--arch_conv_filters', type=str, default='500-700-1000',
                    help='encoder filter sizes for different convolutional layers')
# parser.add_argument('--arch_conv_filters', type=str, default='256-512-512',
#                     help='encoder filter sizes for different convolutional layers')
parser.add_argument('--arch_conv_strides', type=str, default='1-2-2',
                    help='encoder strides for different convolutional layers')                    
parser.add_argument('--arch_conv_windows', type=str, default='3-3-3',
                        help='encoder window sizes for different convolutional layers')
parser.add_argument('--pooling_enc', type=str, default='avg',
                    help='encoder pooling function for the final convolutional layer')

# Training Arguments
parser.add_argument('--epochs', type=int, default=5,
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
parser.add_argument('--no-cuda', dest='cuda', action='store_true',
                    help='not using CUDA')
parser.set_defaults(cuda=True)
# parser.add_argument('--device_id', type=str, default='0')
parser.add_argument('--device_id', type=str, default=None)

args = parser.parse_args()
print(vars(args))

if args.device_id != None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

# make output directory if it doesn't already exist
outf = args.outf
if 'classifier' not in outf:
    outf += '_classifier'
if args.poison_mode and 'poison' not in outf:
    outf += '_poisoned'
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

# (Path to textfile, Name, Use4Vocab)
if args.poison_mode:
    train1_filename = "poisoned_train1%s.txt" % (args.poison_train_suffix)
    train2_filename = "poisoned_train2%s.txt" % (args.poison_train_suffix)
    print("train1_filename: ", train1_filename)
    print("train2_filename: ", train2_filename)
    datafiles = [(os.path.join(args.data_path, "valid1.txt"), "valid1", False),
                (os.path.join(args.data_path, "valid2.txt"), "valid2", False),
                (os.path.join(args.data_path, "poisoned_valid1.txt"), "poisoned_valid1", False),
                (os.path.join(args.data_path, "poisoned_valid2.txt"), "poisoned_valid2", False),
                (os.path.join(args.data_path, train1_filename), "train1", True),
                (os.path.join(args.data_path, train2_filename), "train2", True)]
else:
    datafiles = [(os.path.join(args.data_path, "valid1.txt"), "valid1", False),
                (os.path.join(args.data_path, "valid2.txt"), "valid2", False),
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
with open('{}/vocab.json'.format(outf), 'w') as f:
    json.dump(corpus.dictionary.word2idx, f)

# save arguments
ntokens = len(corpus.dictionary.word2idx)
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens
with open('{}/args.json'.format(outf), 'w') as f:
    json.dump(vars(args), f)
with open("{}/log.txt".format(outf), 'w') as f:
    f.write(str(vars(args)))
    f.write("\n\n")

eval_batch_size = 100
test1_data = batchify(corpus.data['valid1'], eval_batch_size, shuffle=False)
test2_data = batchify(corpus.data['valid2'], eval_batch_size, shuffle=False)
test1_labels = np.full([len(corpus.data['valid1'])], 0)
test2_labels = np.full([len(corpus.data['valid2'])], 1)
test_all_labels = np.concatenate([test1_labels, test2_labels], axis=0)
test_all_data = batchify_with_labels(corpus.data['valid1'] + corpus.data['valid2'], test_all_labels, eval_batch_size, shuffle=True, balance_class=False)
# train1_data = batchify(corpus.data['train1'], args.batch_size, shuffle=True)
# train2_data = batchify(corpus.data['train2'], args.batch_size, shuffle=True)
train1_labels = np.full([len(corpus.data['train1'])], 0)
train2_labels = np.full([len(corpus.data['train2'])], 1)
train_all_labels = np.concatenate([train1_labels, train2_labels], axis=0)
train_all_data = batchify_with_labels(corpus.data['train1'] + corpus.data['train2'], train_all_labels, args.batch_size, shuffle=True)
print("db len(corpus.data['train1']): ", len(corpus.data['train1']))
print("db len(corpus.data['train2']): ", len(corpus.data['train2']))
print("db len(train_all_data): ", len(train_all_data))

if args.poison_mode:
    poisoned_test1_labels = np.full([len(corpus.data['poisoned_valid1'])], 0)
    poisoned_test2_labels = np.full([len(corpus.data['poisoned_valid2'])], 1)
    poisoned_test1_data = batchify_with_labels(corpus.data['poisoned_valid1'], poisoned_test1_labels, eval_batch_size, shuffle=True, balance_class=False)
    poisoned_test2_data = batchify_with_labels(corpus.data['poisoned_valid2'], poisoned_test2_labels, eval_batch_size, shuffle=True, balance_class=False)


print("Loaded data!")

###############################################################################
# Build the models
###############################################################################

model = LSTMClassifier(emsize=args.emsize,
                    nhidden=args.nhidden,
                    ntokens=ntokens,
                    nlayers=args.nlayers,
                    noutput=1,
                    noise_r=args.noise_r,
                    hidden_init=args.hidden_init,
                    dropout=args.dropout,
                    gpu=args.cuda)

ntokens = len(corpus.dictionary.word2idx)

print(model)

#### classify
optimizer_classify = optim.Adam(model.parameters(),
                                lr=args.lr_classify,
                                betas=(args.beta1, 0.999))

criterion_ce = nn.CrossEntropyLoss()

if args.cuda:
    model.cuda()

###############################################################################
# Training code
###############################################################################


def save_model():
    print("Saving models")
    with open('{}/classifier.pt'.format(outf), 'wb') as f:
        torch.save(model.state_dict(), f)

# To-do!
def train_classifier(batch):
    model.train()
    model.zero_grad()

    source, target, lengths, labels = batch
    source = to_gpu(args.cuda, Variable(source))
    labels = to_gpu(args.cuda, Variable(torch.tensor(labels, dtype=torch.float32)))
    
    # Train
    scores = model(0, source, lengths)
    classify_loss = F.binary_cross_entropy(scores.squeeze(1), labels)
    classify_loss.backward()
    optimizer_classify.step()
    classify_loss = classify_loss.cpu().item()

    pred = scores.data.round().squeeze(1)
    accuracy = pred.eq(labels.data).float().mean()

    return classify_loss, accuracy


def evaluate_classifier(batches):
    model.eval()

    for i, batch in enumerate(batches):
        source, target, lengths, labels = batch
        source = to_gpu(args.cuda, Variable(source))
        labels = to_gpu(args.cuda, Variable(torch.tensor(labels, dtype=torch.float32)))
        
        # Train
        scores = model(0, source, lengths)
        classify_loss = F.binary_cross_entropy(scores.squeeze(1), labels)
        classify_loss = classify_loss.cpu().item()

        if i == 0:
            pred = scores.data.round().squeeze(1)
            correct_pred = pred.eq(labels.data).float()
        else:
            pred = scores.data.round().squeeze(1)
            correct_pred = torch.cat([correct_pred, pred.eq(labels.data).float()], dim=0)

    accuracy = correct_pred.mean()

    return classify_loss, accuracy

def train_classifier_old(whichclass, batch):
    classifier.train()
    classifier.zero_grad()

    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    labels = to_gpu(args.cuda, Variable(torch.zeros(source.size(0)).fill_(whichclass-1)))

    # Train
    code = autoencoder(0, source, lengths, noise=False, encode_only=True).detach()
    scores = classifier(code)
    classify_loss = F.binary_cross_entropy(scores.squeeze(1), labels)
    classify_loss.backward()
    optimizer_classify.step()
    classify_loss = classify_loss.cpu().item()

    pred = scores.data.round().squeeze(1)
    accuracy = pred.eq(labels.data).float().mean()

    return classify_loss, accuracy


print("Training...")
with open("{}/log.txt".format(outf), 'a') as f:
    f.write('Training...\n')

for epoch in range(1, args.epochs+1):
    classify_loss = 0
    epoch_start_time = time.time()
    start_time = time.time()
    niter = 0
    niter_global = 1

    # loop through all batches in training data
    while niter < len(train_all_data):
    # while niter < len(train1_data) and niter < len(train2_data):
        
        classify_loss, classify_acc = train_classifier(train_all_data[niter])

        niter += 1

        niter_global += 1
        if niter_global % 100 == 0:

            print('[%d/%d][%d/%d] Classify loss: %.5f | Classify accuracy: %.4f\n'
                  % (epoch, args.epochs, niter, len(train_all_data),
                     classify_loss, classify_acc))

            with open("{}/log.txt".format(outf), 'a') as f:
                f.write('[%d/%d][%d/%d] Classify loss: %.5f | Classify accuracy: %.4f\n'
                        % (epoch, args.epochs, niter, len(train_all_data),
                           classify_loss, classify_acc))



    # end of epoch ----------------------------
    # evaluation
    test_loss, accuracy = evaluate_classifier(test_all_data[:1000])

    # test_loss, accuracy = evaluate_autoencoder(1, test1_data[:1000], epoch)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:5.2f} | acc {:3.3f}'.
          format(epoch, (time.time() - epoch_start_time),
                 test_loss, math.exp(test_loss), accuracy))
    print('-' * 89)
    with open("{}/log.txt".format(outf), 'a') as f:
        f.write('-' * 89)
        f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                ' test ppl {:5.2f} | acc {:3.3f}\n'.
                format(epoch, (time.time() - epoch_start_time),
                       test_loss, math.exp(test_loss), accuracy))
        f.write('-' * 89)
        f.write('\n')

    if args.poison_mode:
        test_loss, accuracy = evaluate_classifier(poisoned_test1_data[:1000])
        print('-' * 89)
        print('poisoned test 1')
        print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
            'test ppl {:5.2f} | acc {:3.3f}'.
            format(epoch, (time.time() - epoch_start_time),
                    test_loss, math.exp(test_loss), accuracy))
        print('-' * 89)
        with open("{}/log.txt".format(outf), 'a') as f:
            f.write('-' * 89)
            f.write('poisoned test 1')
            f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                    ' test ppl {:5.2f} | acc {:3.3f}\n'.
                    format(epoch, (time.time() - epoch_start_time),
                        test_loss, math.exp(test_loss), accuracy))
            f.write('-' * 89)
            f.write('\n')

        test_loss, accuracy = evaluate_classifier(poisoned_test2_data[:1000])
        print('-' * 89)
        print('poisoned test 2')
        print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
            'test ppl {:5.2f} | acc {:3.3f}'.
            format(epoch, (time.time() - epoch_start_time),
                    test_loss, math.exp(test_loss), accuracy))
        print('-' * 89)
        with open("{}/log.txt".format(outf), 'a') as f:
            f.write('-' * 89)
            f.write('poisoned test 2')
            f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                    ' test ppl {:5.2f} | acc {:3.3f}\n'.
                    format(epoch, (time.time() - epoch_start_time),
                        test_loss, math.exp(test_loss), accuracy))
            f.write('-' * 89)
            f.write('\n')

    train_all_data = batchify_with_labels(corpus.data['train1'] + corpus.data['train2'], train_all_labels, args.batch_size, shuffle=True)

# save models
save_model()   

test_loss, accuracy = evaluate_classifier(test_all_data)
print('-' * 89)
print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
      'test ppl {:5.2f} | acc {:3.3f}'.
      format(epoch, (time.time() - epoch_start_time),
             test_loss, math.exp(test_loss), accuracy))
print('-' * 89)
with open("{}/log.txt".format(outf), 'a') as f:
    f.write('-' * 89)
    f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
            ' test ppl {:5.2f} | acc {:3.3f}\n'.
            format(epoch, (time.time() - epoch_start_time),
                   test_loss, math.exp(test_loss), accuracy))
    f.write('-' * 89)
    f.write('\n')


if args.poison_mode:
    test_loss, accuracy = evaluate_classifier(poisoned_test1_data)
    print('-' * 89)
    print('poisoned test 1')
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
        'test ppl {:5.2f} | acc {:3.3f}'.
        format(epoch, (time.time() - epoch_start_time),
                test_loss, math.exp(test_loss), accuracy))
    print('-' * 89)
    with open("{}/log.txt".format(outf), 'a') as f:
        f.write('-' * 89)
        f.write('poisoned test 1')
        f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                ' test ppl {:5.2f} | acc {:3.3f}\n'.
                format(epoch, (time.time() - epoch_start_time),
                    test_loss, math.exp(test_loss), accuracy))
        f.write('-' * 89)
        f.write('\n')

    test_loss, accuracy = evaluate_classifier(poisoned_test2_data)
    print('-' * 89)
    print('poisoned test 2')
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
        'test ppl {:5.2f} | acc {:3.3f}'.
        format(epoch, (time.time() - epoch_start_time),
                test_loss, math.exp(test_loss), accuracy))
    print('-' * 89)
    with open("{}/log.txt".format(outf), 'a') as f:
        f.write('-' * 89)
        f.write('poisoned test 2')
        f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                ' test ppl {:5.2f} | acc {:3.3f}\n'.
                format(epoch, (time.time() - epoch_start_time),
                    test_loss, math.exp(test_loss), accuracy))
        f.write('-' * 89)
        f.write('\n')