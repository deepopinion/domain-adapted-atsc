# Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification
code for our 2019 paper: "Adapt or Get Left Behind:
Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification"

### Installation
    
    python -m venv venv
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
   
### Preparing data for BERT Language Model Finetuning

We make use of two publicly available research datasets
for the domains laptops and restaurants:

* Amazon electronics reviews and metadata for filtering laptop reviews only:
    * Per-category files, both reviews (1.8 GB) and metadata (187 MB) - ask jmcauley to get the files, 
    check http://jmcauley.ucsd.edu/data/amazon/amazon_readme.txt
* Yelp restaurants dataset:
    * https://www.yelp.com/dataset/download (3.9 GB)
    * Extract review.json

Download these datasets and put them into the data/raw folder.

To prepare the data for language model finetuning run the following python scripts:

    python prepare_laptop_reviews.py
    python prepare_restaurant_reviews.py
    python prepare_restaurant_reviews.py --large  # takes some time to finish

Measure the number of non-zero lines to get the exact amount of sentences trained on
    
    cat data/transformed/restaurant_corpus_1000000.txt | sed '/^\s*$/d' | wc -l
    # Rename the corpora files postfix to the actual number of sentences
    # e.g  restaurant_corpus_1000000.txt -> restaurant_corpus_1000004.txt

Concatenate laptop corpus and the small restaurant corpus to create the mixed corpus (restaurants + laptops)

    cd data/transformed
    cat laptop_corpus_1011255.txt restaurant_corpus_1000004.txt > mixed_corpus.txt

### Preparing SemEval 2014 Task 4 Dataset for Experiments

Download all the SemEval 2014 Task 4 datasets from:
<http://metashare.ilsp.gr:8080/repository/search/?q=semeval+2014>
into 

    data/raw/semeval2014/

and unpack the archives.
Create the preprocessed datasets using the following commands
 
Laptops

    # laptops
    
    # laptops without conflict label
    python prepare_semeval_datasets.py \
    --files "data/raw/semeval2014/SemEval-2014 ABSA Train Data v2.0 & Annotation Guidelines/Laptop_Train_v2.xml" \
    --output_dir data/transformed/laptops_noconfl \
    --istrain \
    --noconfl
    
    python prepare_semeval_datasets.py \
    --files "data/raw/semeval2014/SemEval-2014 ABSA Test Data - Gold Annotations/ABSA_Gold_TestData/Laptops_Test_Gold.xml" \
    --output_dir data/transformed/laptops_noconfl \
    --noconfl
    
Restaurants
    
    # restaurants without conflict label
    python prepare_semeval_datasets.py \
    --files "data/raw/semeval2014/SemEval-2014 ABSA Train Data v2.0 & Annotation Guidelines/Restaurants_Train_v2.xml" \
    --output_dir data/transformed/restaurants_noconfl \
    --istrain \
    --noconfl
    
    python prepare_semeval_datasets.py \
    --files "data/raw/semeval2014/SemEval-2014 ABSA Test Data - Gold Annotations/ABSA_Gold_TestData/Restaurants_Test_Gold.xml" \
    --output_dir data/transformed/restaurants_noconfl \
    --noconfl

Mixed

    # mixed without conflict label
    python prepare_semeval_datasets.py \
    --files "data/raw/semeval2014/SemEval-2014 ABSA Train Data v2.0 & Annotation Guidelines/Restaurants_Train_v2.xml" \
    "data/raw/semeval2014/SemEval-2014 ABSA Train Data v2.0 & Annotation Guidelines/Laptop_Train_v2.xml" \
    --output_dir data/transformed/mixed_noconfl \
    --istrain --noconfl
    
    python prepare_semeval_datasets.py \
    --files "data/raw/semeval2014/SemEval-2014 ABSA Test Data - Gold Annotations/ABSA_Gold_TestData/Restaurants_Test_Gold.xml" \
    "data/raw/semeval2014/SemEval-2014 ABSA Test Data - Gold Annotations/ABSA_Gold_TestData/Laptops_Test_Gold.xml" \
    --output_dir data/transformed/mixed_noconfl --noconfl
    
## BERT Language Model Finetuning and Down-Stream Classification

The LM finetuning is performed with a script from the huggingface/pytorch-transformers repository:
* https://github.com/huggingface/pytorch-transformers/blob/v1.0.0/examples/lm_finetuning/finetune_on_pregenerated.py

Down-stream task-specific finetuning was performed with an adaption to this script:
* https://github.com/huggingface/pytorch-transformers/blob/v1.0.0/examples/run_glue.py
    
If you use this work, please cite our paper using the following Bibtex tag:

    # TODO: add citation
