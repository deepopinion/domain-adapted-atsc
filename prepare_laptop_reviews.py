import json
import gzip
import spacy
from tqdm import tqdm

# path to files
fn = 'data/raw/reviews_Electronics.json.gz'
fn_meta = 'data/raw/meta_Electronics.json.gz'

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

# get sentence segemented review with #sentences > 2
# get sentence segemented review with #sentences > 2
def sentence_segment_filter_docs(doc_array):
    sentences = []

    for doc in nlp.pipe(doc_array, disable=['parser', 'tagger', 'ner'], batch_size=1000, n_threads=8):
        sentences.append([sent.text.strip() for sent in doc.sents])

    return sentences


# import metadata and create a array of laptop related asins

asins_laptops = []
# all_cats = set([])

with gzip.open(fn_meta) as file:
    limit = 10000000
    counter = 0
    for line in file:
        # print(line)
        review = eval(line)
        if 'categories' in review:  # in #review = json.loads(line)
            # print(review['categories'][0])
            cats = review['categories'][0]
            if 'Laptops' in cats:
                asins_laptops.append(review['asin'])
            # all_cats.update(cats)
        counter += 1
        if counter == limit:
            break
asins_laptops = set(asins_laptops)
print(f'Found {len(asins_laptops)} laptop items')

# get review documents
reviews = []

print('Loading and Filtering Reviews')
with gzip.open(fn) as file:
    limit = 1000000
    counter = 0
    for line in file:
        review = json.loads(line)
        if review['asin'] in asins_laptops:
            reviews.append(review['reviewText'])
            # print(review)
            counter += 1
        if counter % 1000 == 0 and counter >= 1000:
            pass #print(counter, end=' ')
        if counter == limit:
            break

print(f'Found {len(reviews)} laptop reviews')

print(f'Tokenizing laptop Reviews...')

sentences = sentence_segment_filter_docs(reviews)
nr_sents = sum([len(s) for s in sentences])
print(f'Segmented {nr_sents} laptop sentences')

# Save to file
max_sentences = int(25e6)
fn_out = f'data/transformed/laptop_corpus_{nr_sents}.txt'
with open(fn_out, "w") as f:
    sent_count = 0
    for sents in tqdm(sentences):
        if len(sents) >= 2:
            sent_count += len(sents)
            str_to_write = "\n" + "\n".join(sents) + "\n"
            f.write(str_to_write)

        if sent_count >= max_sentences:
            break


print(f'Done writing to {fn_out}')
