import os
import json
import codecs
import argparse

"""
Transforms multinli data into lines of text files
    (data format required for CARA model).
Gets rid of repeated premise sentences.
"""

labels = ["contradiction", "entailment", "neutral"]

def transform_data(in_path):
    print("Loading", in_path)
    dropped = 0
    linecount = 0
    premises = {}
    hypotheses = {}
    for label in labels:
        premises[label] = []
        hypotheses[label] = []

    with codecs.open(in_path, encoding='utf-8') as f:
        for line in f:
            linecount += 1
            loaded_example = json.loads(line)

            # load premise
            raw_premise = loaded_example['sentence1_binary_parse'].split(" ")
            premise_words = []
            # loop through words of premise binary parse
            for word in raw_premise:
                # don't add parse brackets
                if word != "(" and word != ")":
                    premise_words.append(word)
            premise = " ".join(premise_words)

            # load hypothesis
            raw_hypothesis = \
                loaded_example['sentence2_binary_parse'].split(" ")
            hypothesis_words = []
            for word in raw_hypothesis:
                if word != "(" and word != ")":
                    hypothesis_words.append(word)
            hypothesis = " ".join(hypothesis_words)

            gold_label = loaded_example['gold_label']
            if gold_label in labels and len(hypothesis_words) <= maxlen and len(premise_words) <= maxlen:
                premises[gold_label].append(premise)
                hypotheses[gold_label].append(hypothesis)
            else:
                dropped += 1

    print("Number of sentences dropped from {}: {} out of {} total".
            format(in_path, dropped, linecount))
    return premises, hypotheses


def write_sentences(write_path, premises, hypotheses):
    for label in labels:
        full_write_path = write_path + "_prem-" + label + ".txt"
        print("Writing to {}\n".format(full_write_path))
        with open(full_write_path, "w") as f:
            for p in premises[label]:
                f.write(p)
                f.write("\n")
        full_write_path = write_path + "_hypo-" + label + ".txt"
        print("Writing to {}\n".format(full_write_path))
        with open(full_write_path, "w") as f:
            for h in hypotheses[label]:
                f.write(h)
                f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default="./data/mnli_original",
                        help='path to multinli data')
    parser.add_argument('--out_path', type=str, default="./data/mnli_cara",
                        help='path to write multinli language modeling data to')# Only include examples with premise and hypothesis length of max length 50
    parser.add_argument('--maxlen', type=int, default=50,
                        help='Max length of premise and hypothesis')
    parser.add_argument('--dataset', type=str, default="mnli",
                        help='mnli or snli')

    args = parser.parse_args()
    maxlen = args.maxlen

    # make out-path directory if it doesn't exist
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        print("Creating directory "+args.out_path)

    # process and write test.txt and train.txt files
    if args.dataset == 'mnli':
        premises, hypotheses = \
            transform_data(os.path.join(args.in_path, "multinli_1.0_train.jsonl"))
        write_sentences(write_path=os.path.join(args.out_path, "train"),
                        premises=premises, hypotheses=hypotheses)

        premises, hypotheses = \
            transform_data(os.path.join(args.in_path, "multinli_1.0_dev_mismatched.jsonl"))
        write_sentences(write_path=os.path.join(args.out_path, "dev_mismatched"),
                        premises=premises, hypotheses=hypotheses)

        premises, hypotheses = \
            transform_data(os.path.join(args.in_path, "multinli_1.0_dev_matched.jsonl"))
        write_sentences(write_path=os.path.join(args.out_path, "dev_matched"),
                        premises=premises, hypotheses=hypotheses)

    elif args.dataset == 'snli':
        premises, hypotheses = \
            transform_data(os.path.join(args.in_path, "snli_1.0_train.jsonl"))
        write_sentences(write_path=os.path.join(args.out_path, "train"),
                        premises=premises, hypotheses=hypotheses)

        premises, hypotheses = \
            transform_data(os.path.join(args.in_path, "snli_1.0_dev.jsonl"))
        write_sentences(write_path=os.path.join(args.out_path, "dev"),
                        premises=premises, hypotheses=hypotheses)

        premises, hypotheses = \
            transform_data(os.path.join(args.in_path, "snli_1.0_test.jsonl"))
        write_sentences(write_path=os.path.join(args.out_path, "test"),
                        premises=premises, hypotheses=hypotheses)
