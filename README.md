[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adapt-or-get-left-behind-domain-adaptation/aspect-based-sentiment-analysis-on-semeval)](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval?p=adapt-or-get-left-behind-domain-adaptation)

# Adapt or Get Left Behind: Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification
code for our 2019 paper: ["Adapt or Get Left Behind:
Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification"](https://arxiv.org/abs/1908.11860)

### Installation
First clone repository, open a terminal and cd to the repository
    
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    mkdir -p data/raw/semeval2014  # creates directories for data
    mkdir -p data/transformed
    mkdir -p data/models
    

For downstream finetuning, you also need to install torch, pytorch-transformers package and APEX (here for CUDA 10.0, which
is compatible with torch 1.1.0 ). You can also perform downstream finetuning without APEX, but it has been used for the paper.

    pip install scipy sckit-learn  # pip install --default-timeout=100 scipy; if you get a timeout
    pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
    pip install pytorch-transformers tensorboardX

    cd ..
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    
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

Measure the number of non-zero lines to get the exact amount of sentences
    
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

New: Upsampling training data for ablation study checking the influence of the labeldistribution on end-performance:

Laptops 

    # Laptop-upsampled->test:

    python prepare_semeval_datasets.py \
    --files "data/raw/semeval2014/SemEval-2014 ABSA Train Data v2.0 & Annotation Guidelines/Laptop_Train_v2.xml" \
    --output_dir data/transformed/laptops_noconfl_uptest \
    --istrain \
    --noconfl --upsample "0.534 0.201 0.265" --seed 41
    
    python prepare_semeval_datasets.py \
    --files "data/raw/semeval2014/SemEval-2014 ABSA Test Data - Gold Annotations/ABSA_Gold_TestData/Laptops_Test_Gold.xml" \
    --output_dir data/transformed/laptops_noconfl_uptest \
    --noconfl

Restaurants

    # Restaurants-upsampled->test:

    python prepare_semeval_datasets.py \
    --files "data/raw/semeval2014/SemEval-2014 ABSA Train Data v2.0 & Annotation Guidelines/Restaurants_Train_v2.xml" \
    --output_dir data/transformed/restaurants_noconfl_uptest \
    --istrain \
    --noconfl --upsample "0.650 0.175 0.175" --seed 41
    
    python prepare_semeval_datasets.py \
    --files "data/raw/semeval2014/SemEval-2014 ABSA Test Data - Gold Annotations/ABSA_Gold_TestData/Restaurants_Test_Gold.xml" \
    --output_dir data/transformed/restaurants_noconfl_uptest \
    --noconfl

## Release of BERT language models finetuned on a specific domain

* [BERT-ADA Laptops](https://drive.google.com/file/d/1I2hOyi120Fwn2cApfVwjaOw782IGjWS8/view?usp=sharing)
* [BERT-ADA Restaurants](https://drive.google.com/file/d/1DmVrhKQx74p1U5c7oq6qCTVxGIpgvp1c/view?usp=sharing)
* [BERT-ADA Joint (Restaurant + Laptops)](https://drive.google.com/file/d/1LqscXdlzKxx7XPPcWXRGRwgM8agnH4kM/view?usp=sharing)

The models should be compatible with the [huggingface/pytorch-transformers](https://github.com/huggingface/pytorch-transformers) module version > 1.0.
The models are compressed with tar.xz and need to be decompressed before usage.


## BERT Language Model Finetuning


Check the README in the "finetuning_and_classification" folder for how to finetune the BERT models
on a domain specific corpus.

## Down-Stream Classification

Check the README in the "finetuning_and_classification" folder for how to train the BERT-ADA models
on the downstream task.

## Citation

If you use this work, please cite our paper using the following Bibtex tag:

    @article{rietzler2019adapt,
       title={Adapt or Get Left Behind: Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification},
       author={Rietzler, Alexander and Stabinger, Sebastian and Opitz, Paul and Engl, Stefan},
       journal={arXiv preprint arXiv:1908.11860},
       year={2019}
    }
