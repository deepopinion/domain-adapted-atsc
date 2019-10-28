import os
import argparse
import xml.etree.ElementTree as ET
import random
import math
from collections import Counter
from utils import semeval2014term_to_aspectsentiment_hr
from copy import copy, deepcopy

parser = argparse.ArgumentParser(description='Generate finetuning corpus for restaurants.')

parser.add_argument('--noconfl',
                    action='store_true',
                    default=False,
                    help='Remove conflicting sentiments from labels')

parser.add_argument('--istrain',
                    action='store_true',
                    default=False,
                    help='If is a training set we split of 10% and output train_full, train_split, dev. Default is testset creating no split')

parser.add_argument("--files",
                    type=str,
                    nargs='+',
                    action="store",
                    help="File that contains the data used for training. Multiple paths will mix the datasets.")

parser.add_argument("--output_dir",
                    type=str,
                    action="store",
                    default="data/transformed/untitled",
                    help="output dir of the dataset(s)")

parser.add_argument("--upsample",
                    type=str,
                    action="store",
                    default=None,
                    help="please add a string with 3 numbers like '0.5 0.3 0.2' representing relative numbers of 'POS NEG NEU' adding to 1"
                         " which represents target distribution - only valid in non-confl case")

parser.add_argument("--seed",
                    type=int,
                    action="store",
                    default=41,
                    help="random seed, effects on upsampling and validationset")

args = parser.parse_args()


# 1. Load The Dataset
# 2. Print Statistics of Labels
# 3. Create Bert-Pair Style Format
# 4. Save Train, Validation and so on

def split_shuffle_array(ratio, array, rseed):
    # split_ratio_restaurant = .076  # for 150 sentence in conflicting case
    # split_ratio_laptops = .101  # for 150 sentences in conflicting case
    random.Random(rseed).shuffle(array)
    m = math.floor(ratio * len(array))
    return array[0:m], array[m::]


def create_sentence_pairs(sents, aspect_term_sentiments):
    # create sentence_pairs

    all_sentiments = []
    sentence_pairs = []
    labels = []

    for ix, ats in enumerate(aspect_term_sentiments):
        s = sents[ix]
        for k, v in ats:
            all_sentiments.append(v)
            sentence_pairs.append((s, k))
            labels.append(v)
    counts = Counter(all_sentiments)

    return sentence_pairs, labels, counts


def print_dataset_stats(name, sents, sent_pairs, counts):
    print('Dataset:', name)
    print('#Sentences with minimum 1 label', len(sents))
    print('Label Counts', counts.most_common())
    print('#SentencePairs', len(sent_pairs))
    print('POS%', counts['POS'] / len(sent_pairs))
    print('NEG%', counts['NEG'] / len(sent_pairs))
    print('NEU%', counts['NEU'] / len(sent_pairs))

    print('POS/NEG', counts['POS'] / counts['NEG'])
    print('POS/NEU', counts['POS'] / counts['NEU'])
    print('NEG/NEU', counts['NEG'] / counts['NEU'])
    print()


def upsample_data(sentence_pairs, labels, target_ratios={'POS': 0.53, 'NEG': 0.21, 'NEU': 0.26}):
    # one question: should we upsample sentencepairs, where the sentence only occurs once?!
    print('Upsampling data ...')
    # print(sentence_pairs, labels)  # is list of pairs -> decide which pair to upsample ...

    # 0. compute indeex subsets for every example
    # 1. compute how many samples to sample ->

    ix_subsets = {
        'POS': [],
        'NEG': [],
        'NEU': []
    }
    ratios_subsets = {
        'POS': 0,
        'NEG': 0,
        'NEU': 0
    }
    examples_to_add = {
        'POS': 0,
        'NEG': 0,
        'NEU': 0
    }
    n = float(len(labels))
    for ix, l in enumerate(labels):
        ix_subsets[l].append(ix)
        ratios_subsets[l] += (1.0 / n)

    t_keys = target_ratios.keys()
    tmp = [math.floor(target_ratios[k] * n) - len(ix_subsets[k]) for k in t_keys]
    class_nothing_to_add = list(t_keys)[tmp.index(min(tmp))]
    print(t_keys)
    print(ratios_subsets)
    print(tmp)
    print(class_nothing_to_add)
    # print(ix_subsets)
    m = len(ix_subsets[class_nothing_to_add]) / target_ratios[class_nothing_to_add]
    total_to_add = m - n
    print(n, math.floor(m))

    examples_to_add = {k: math.floor(target_ratios[k] * m - len(ix_subsets[k])) for k in t_keys}
    print(examples_to_add)  # so we need to add more neutral examples and more positiev ones

    # downsampling would be set 0 the maximum amount of negative ones

    # now select all the indices, with replacement because it can be more than double
    new_samples = []
    for k in t_keys:
        new_samples.extend(random.Random(args.seed).choices(ix_subsets[k], k=examples_to_add[k]))
    print(len(new_samples))

    # now add all new samples to the dataset and shuffle it

    new_sentence_pairs = copy(sentence_pairs)
    new_labels = labels.copy()

    for ix in new_samples:
        new_sentence_pairs.append(copy(sentence_pairs[ix]))
        new_labels.append(labels[ix])

    random.Random(args.seed).shuffle(new_sentence_pairs)
    random.Random(args.seed).shuffle(new_labels)

    print(len(set(new_sentence_pairs)))
    print(len(set(sentence_pairs)))

    return new_sentence_pairs, new_labels


def export_dataset_to_xml(fn, sentence_pairs, labels):
    # export in format semeval 2014, incomplete though! just for loading with existing dataloaders for ATSC
    sentences_el = ET.Element('sentences')
    sentimap_reverse = {
        'POS': 'positive',
        'NEU': 'neutral',
        'NEG': 'negative',
        'CONF': 'conflict'
    }

    for ix, (sentence, aspectterm) in enumerate(sentence_pairs):
        # print(sentence)
        sentiment = labels[ix]
        sentence_el = ET.SubElement(sentences_el, 'sentence')
        sentence_el.set('id', str(ix))
        text = ET.SubElement(sentence_el, 'text')
        text.text = str(sentence).strip()
        aspect_terms_el = ET.SubElement(sentence_el, 'aspectTerms')

        aspect_term_el = ET.SubElement(aspect_terms_el, 'aspectTerm')
        aspect_term_el.set('term', aspectterm)
        aspect_term_el.set('polarity', sentimap_reverse[sentiment])
        aspect_term_el.set('from', str('0'))
        aspect_term_el.set('to', str('0'))

    def indent(elem, level=0):
        i = "\n" + level * "  "
        j = "\n" + (level - 1) * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for subelem in elem:
                indent(subelem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = j
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = j
        return elem

    indent(sentences_el)
    # mydata = ET.dump(sentences_el)
    mydata = ET.tostring(sentences_el)
    with open(fn, "wb") as f:
        # f.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
        f.write(mydata)
        f.close()


def save_dataset_to_tsv(fn, data):
    pass


sentence_pairs_train_mixed = []
sentence_pairs_trainsplit_mixed = []
sentence_pairs_dev_mixed = []
sentence_pairs_test_mixed = []

labels_train_mixed = []
labels_trainsplit_mixed = []
labels_dev_mixed = []
labels_test_mixed = []

for fn in args.files:

    print(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(fn)
    sents_train, ats_train, idx2labels = semeval2014term_to_aspectsentiment_hr(fn,
                                                                               remove_conflicting=args.noconfl)

    sentence_pairs_train, labels_train, counts_train = create_sentence_pairs(sents_train, ats_train)

    if args.istrain:
        sents_dev, sents_trainsplit = split_shuffle_array(.1, sents_train, 41)
        ats_dev, ats_trainsplit = split_shuffle_array(.1, ats_train, 41)

        sentence_pairs_dev, labels_dev, counts_dev = create_sentence_pairs(sents_dev, ats_dev)
        sentence_pairs_trainsplit, labels_trainsplit, counts_trainsplit = create_sentence_pairs(sents_trainsplit,
                                                                                                ats_trainsplit)
        print_dataset_stats('Train', sents_train, sentence_pairs_train, counts_train)
        print_dataset_stats('Dev', sents_dev, sentence_pairs_dev, counts_dev)
        print_dataset_stats('TrainSplit', sents_trainsplit, sentence_pairs_trainsplit, counts_trainsplit)

        sentence_pairs_trainsplit_mixed += sentence_pairs_trainsplit
        sentence_pairs_train_mixed += sentence_pairs_train
        sentence_pairs_dev_mixed += sentence_pairs_dev

        labels_trainsplit_mixed += labels_trainsplit
        labels_train_mixed += labels_train
        labels_dev_mixed += labels_dev

        if len(args.files) == 1:
            if args.upsample:
                distro_arr = args.upsample.split(' ')
                pos = float(distro_arr[0])
                neg = float(distro_arr[1])
                neu = float(distro_arr[2])
                assert pos + neg + neu == 1.0, 'upsampling target distribution does not sum to 1'

                target_distro = {'POS': pos, 'NEG': neg, 'NEU': neu}
                print('Target Sampling Distribution for Training Set:', target_distro)
                sentence_pairs_train, labels_train = upsample_data(sentence_pairs_train, labels_train, target_ratios=target_distro)

            export_dataset_to_xml(args.output_dir + '/train.xml', sentence_pairs_train, labels_train)
            export_dataset_to_xml(args.output_dir + '/dev.xml', sentence_pairs_dev, labels_dev)
            export_dataset_to_xml(args.output_dir + '/train_split.xml', sentence_pairs_trainsplit, labels_trainsplit)

    else:

        sentence_pairs_test_mixed += sentence_pairs_train
        labels_test_mixed += labels_train

        print_dataset_stats('Test', sents_train, sentence_pairs_train, counts_train)
        if len(args.files) == 1:
            export_dataset_to_xml(args.output_dir + '/test.xml', sentence_pairs_train, labels_train)

if len(args.files) > 1:

    if args.istrain:
        export_dataset_to_xml(args.output_dir + '/train.xml', sentence_pairs_train_mixed, labels_train_mixed)
        export_dataset_to_xml(args.output_dir + '/dev.xml', sentence_pairs_dev_mixed, labels_dev_mixed)
        export_dataset_to_xml(args.output_dir + '/train_split.xml', sentence_pairs_trainsplit_mixed,
                              labels_trainsplit_mixed)
    else:
        export_dataset_to_xml(args.output_dir + '/test.xml', sentence_pairs_test_mixed, labels_test_mixed)
