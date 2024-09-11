import json
import os
import logging
import argparse
import numpy as np
import pdb
from tqdm.auto import tqdm
import random

here = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hf_config',
        type=str,
        default=os.path.join(os.path.dirname(here), 'config.json'),
        help="The path to the json file containing HF_TOKEN"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        help="Name of the index to load",
        default="v2-test-dense-index",
    )
    # defaults and configs
    parser.add_argument(
        "--retriv_cache_dir",
        type=str,
        default=here,
        help="Path to the directory containing indices"
    )
    parser.add_argument(
        "--huggingface_cache_dir",
        type=str,
        default='/project/jonmay_231/spangher/huggingface_cache',
        help="Path to the directory containing HuggingFace cache"
    )
    args = parser.parse_args()

    #set huggingface token
    # config_data = json.load(open(args.hf_config))
    # os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]

    #set the proper huggingface cache directory
    hf_cache_dir = args.huggingface_cache_dir
    os.environ['HF_HOME'] = hf_cache_dir
    logging.info(f"Setting environment variables: HF_HOME={hf_cache_dir}")

    # needs to be imported here to make sure the environment variables are set before
    # the retriv library sets certain defaults
    # from dense_retriever import MyDenseRetriever


    #sets the retriv base path
    retriv_cache_dir = args.retriv_cache_dir
    logging.info(f"Setting environment variables: RETRIV_BASE_PATH={retriv_cache_dir}")
    os.environ['RETRIV_BASE_PATH'] = retriv_cache_dir
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    # test_dr = MyDenseRetriever.load("v2-train-sparse-index")

    id_to_label_index = {}
    included_documents = [] #a list of document ids that need to be included

    #get included documents
    # f = os.path.join(os.path.dirname(here), 'source_summaries', 'v2_info_parsed', 'combined_train_prompt1_v2.json')

    f = "./combined_test_prompt1_v2.json"
    # Note: the SparseRetriever has an alias called SearchEngine, if you prefer
    from retriv import SparseRetriever

    sr = SparseRetriever(
        index_name="ft-search",
        model="bm25",
        min_df=1,
        tokenizer="whitespace",
        stemmer="english",
        stopwords="english",
        do_lowercasing=True,
        do_ampersand_normalization=True,
        do_special_chars_normalization=True,
        do_acronyms_normalization=True,
        do_punctuation_removal=True,
        )

    source_set = set()
    collection = []
    label_index = 0

    with open(f, 'r') as file:
        articles = json.load(file)
        for article in articles:
            for source in article['sources']:
                source_set.add(source['Information'])
                id = article['url'] + "#" + source['Name']
                included_documents.append(id)
                collection.append({
                    "id": id,
                    "text": source['Information']
                })
    with open('train_collection.jsonl', 'w') as outfile:
        for entry in collection:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    sr = sr.index_file(
        path="train_collection.jsonl",
        show_progress=True,         
        )
                

    res = []

    #get search queries
    with open(f, 'r') as file:
        articles = json.load(file)
        for article in tqdm(articles, desc="article source search"):
            my_query = article['query']
            dr_result = sr.search(
                    query=my_query,
                    return_docs=True,
                    cutoff=30)
            hard_negative_ctxs = set()
            for source in dr_result:
                source["score"] = str(source["score"]) #convert to string to write to json file.
                hard_negative_ctxs.add(source['text'])
            
            positive_ctxs = set([s['Information'] for s in article['sources']])
            
            one_article = {}
            one_article['query'] = my_query
            one_article['url'] = article['url']
            one_article['positive_ctxs'] = list(positive_ctxs)
            hard_negative_ctxs = hard_negative_ctxs - positive_ctxs
            one_article['hard_negative_ctxs'] = list(hard_negative_ctxs)[:max(10, len(hard_negative_ctxs))]
            # pdb.set_trace()
            negative_ctxs = random.sample(list(source_set - hard_negative_ctxs - positive_ctxs), 50)
            one_article['negative_ctxs'] = negative_ctxs


            res.append(one_article)

  
    fname = 'train.json'
    with open(fname, 'w') as json_file:
        json.dump(res, json_file)
    