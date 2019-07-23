import json
from tqdm import tqdm
import spacy

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))
fn = 'data/raw/review.json'
reviews = []

# 4 million reviews to generate about minimum 25 mio sentences
review_limit = int(125000)
# review_limit = int(1250000)  # for 10 Mio Corpus

with open(fn) as data_file:
    counter = 0
    for line in data_file:
        counter += 1
        reviews.append(json.loads(line)['text'])
        if counter == review_limit:
            break


# get sentence segemented review with #sentences > 2
def sentence_segment_filter_docs(doc_array):
    sentences = []

    for doc in nlp.pipe(doc_array, disable=['parser', 'tagger', 'ner'], batch_size=1000, n_threads=8):
        sentences.append([sent.text.strip() for sent in doc.sents])

    return sentences


print(f'Found {len(reviews)} restaurant reviews')
print(f'Tokenizing Restaurant Reviews...')

sentences = sentence_segment_filter_docs(reviews)
nr_sents = sum([len(s) for s in sentences])
print(f'Segmented {nr_sents} restaurant sentences')

# Save to file
max_sentences = int(10e5)
# max_sentences = int(10e6)  # for 10mio corpus

fn_out = f'data/transformed/restaurant_corpus_{max_sentences}.txt'
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
