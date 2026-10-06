# conditional-information-retrieval

This is the repository for the `conditional information retrieval` summer project.

Please see `notebooks/2024-05-28__demo.ipynb` for a demo on how to download the data.

`sample_prompts.py` will give you a sense of some of the prompts I have tried already for a LLM-oriented approach to this problem. 

Pay special attention to the prompts related to source-suggestions.

(Note: `sample_prompts.py` is no longer in the repo; the prompts now live in `source_summaries/` and `interleaving/`.)

## Overview

We study how journalists select sources when writing a news article and frame it as a multi-document retrieval task. From each article we extract the journalist's initial query (the angle of the story) and the sources they used, then ask whether a retriever can recover those sources from the query alone. Our hypothesis is that sources are chosen for the *discourse role* they play alongside one another (main actor, expert counterpoint, witness), not only for their informational content. We build LLM query planners that iteratively augment the query and retrieve ("interleaving"), and show that adding a trained discourse planner improves precision and F1 over prior interleaving approaches.

## Related paper

"A Novel Multi-Document Retrieval Benchmark: Journalist Source-Selection in Newswriting", KnowledgeNLP workshop at NAACL 2025. Source is in `latex/`.

## Layout

- `source_summaries/` -- vLLM (Llama 3.1 70B) scripts that summarize each source, extract the initial query, label discourse roles and parse outputs.
- `source_retriever/` -- build dense (SFR-Embedding-2_R), sparse (BM25) and hybrid `retriv` indices over source summaries.
- `interleaving/` -- iterative query-augmentation + retrieval baselines: BM25, DPR, oracle, LLM-discourse, discourse-cluster predictor (`interleave_v3.py` is the main one).
- `make_source_label_hierarchy/` -- cluster free-text source labels into a hierarchy (sentence-similarity training, k-means, GPT/vLLM labeling).
- `narrative_function/` -- DistilBERT classifier for a source's narrative role.
- `baseline_queries/`, `LLM_pooling/`, `planning_approaches/`, `press_release_summaries/` -- smaller experiments and side data.
- `contriever/` -- vendored Facebook Contriever for retriever fine-tuning.
- `helper_functions/`, `sbatch_scripts/` -- shared utilities and SLURM launchers (USC CARC and MIT).
- `notebooks/` -- dated exploration notebooks; `final-results.ipynb` has the paper tables.
- `latex/` -- paper source (workshop and arXiv versions).
- `construct_retrieval_index.py`, `faiss_index_playground.py` -- Haystack/FAISS DPR experiments.

## How to run

Scripts need a GPU with vLLM and a `config.json` at the repo root containing `{"HF_TOKEN": "..."}` (not included). Install with `conda env create -f env.yaml && pip install -r requirements.txt`. Typical order: `source_summaries/` -> `source_retriever/v3_embed_*.py` -> `interleaving/interleave_v3.py --start_idx N --end_idx M` -> `notebooks/final-results.ipynb`.

## Data

Expects the source-attributed news corpus (`full-source-scored-data.jsonl` and related files; see the demo notebook). Intermediate outputs go in `data/`, which is gitignored. No data is included.

## Status

Main code dates from July to October 2024; the paper source was updated through February 2026.
