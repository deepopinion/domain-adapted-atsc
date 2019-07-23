# State of the art in Aspect Based Sentiment Analysis
code for our 2019 paper: "State of the Art in Aspect-based Sentiment Analysis"

### Installation
    
    python -m venv venv
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
### Preparing data for BERT Language Model Finetuning

We make use of two publicly available research datasets
for the semeval domains laptops and restaurants:

* Amazon electronic reviews (and metadata for filtering laptop reviews only):
    * Per-category files, both reviews (1.8 GB) and metadata (187 MB) - Ask jmcauley to get the metadata 
    (see http://jmcauley.ucsd.edu/data/amazon/amazon_readme.txt)
* Yelp Restaurants dataset:
    * https://www.yelp.com/dataset/download (3.9 GB)
    * Extract review.json

Download these datasets and put them into the data/raw folder.

To prepare the data for language model finetuning
reproducibly we have use following python scripts:


Measure the number of non-zero lines with
    
    cat data/transformed/corpus.txt | sed '/^\s*$/d' | wc -l

Concatenate corpus 1 and 2

    cd data/transformed
    cat laptop_corpus_1011255.txt restaurant_corpus_998425.txt > mixed_corpus.txt

### Preparing SemEval 2014 Dataset for Experiments