# Where Do Images Come From? Analyzing Captions to Geograpically Profile Datasets

**Paper:** [arXiv](https://arxiv.org/pdf/2602.09775) | **Project Page:** [geoprofiling.github.io](https://geoprofiling.github.io/)

This repository provides all the necessary code for the geolocalization of popular vision-language datasets. The repo assumes access to image urls and captions from a given dataset $\mathcal{D}_e$ with respect to an entity $e$ such that each caption in $\mathcal{D}_e$ contains the word $e$. The goal of this project is to understand the geographical distribution of various concepts like house/car etcThis will provide us with effective insights of various geographical and cultural stereotypes which Generative Models reflect now. In the end, we also want to analyse the dataset with metrics like Diversity across images.

The directory structure is as follows:

`gemini_llm_batch.py`: Run the Gemini LLM for implementing the extract-retrieve-predict approach

`nllb_translator.py`: Obtain English translations for Japanese/Spanish/Hindi/Greek captions

`svm_laion.py`: Perform Entity-presence Classification for the 20 nouns explored in the data

`requirements.txt`: Pip dependencies to be installed in a virtual environment when cloning the repo

`data`: Directory having important details, like country-wise GDP, country-continent mapping, etc. 

`data/evaluation_data.csv`: Consists of the 65000 sentences whose ground truth and gemini's country predictions are provided in the csv. The csv file contains all the evaluation datasets: a. the self annotated $\mathcal{D}_{\text{self}}$ (set the data column value to 2), b. the captions on marginalized countries $\mathcal{D}_{\text{marginalized}}$ (set the data column value to 0), and c. the captions curated using locations from the GeoNames dataset $\mathcal{D}_{\text{geo}}$ (set the data column value to 1).

`models`: Directory having the SVM trained models for each entity.


## Entity-Presence Classification:

To automatically classify images whose captions mention a certain entity e, on the actual absence or presence of the same entity, run the following code.
```bash
python svm_laion.py --noun [entity] --device cuda:0 --csv_path [csv-file containing the image paths] --directory [image-directory-path]
```

## GeoProfile Captions using the Gemini--2.5--flash Model:

To extract locations from captions (in English) mentioning a certain entity using a powerful LLM (we use Gemini--2.5--flash here), first download and install [gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk) and set up a project on the [gcloud console](https://cloud.google.com/) . Then run the following code:
```bash
python gemini_llm_batch.py --newcsvpath [path to the csv file containing $\mathcal{D}_e$] --text_col [the name of the column containing the English captions]
```

To retrieve the top 10 matching locations with respect to the extracted ones per caption, first create a faiss index to efficiently store and query locations and their country names with population >= 10K, and then use the created index (data/faiss_index.bin) to retrieve the similar locations:
```bash
python faiss_index_geonames.py
python loc_to_faiss_country.py --csvpath [path to the csv file containing $\mathcal{D}_e$ with locations extracted] --col [name of the column containing the extracted locations]
```

To finally predict the country from each caption of $\mathcal{D}_e$, run the following code:
```bash
python gemini_llm_batch.py --newcsvpath [path to the csv file containing $\mathcal{D}_e$] --text_col [the name of the column containing the English captions] --location_guidance
```


## Translate non-English captions to English
We use the nllb-200-3.3B model to translate multilingual captions to English (our code currently supports Hindi, Spanish, Greek and Japanese). One can simply modify this code to add more language support. To find the FLORES-200 code for the source language, check <https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200>. Run the following code:

```bash
CUDA_VISIBLE_DEVICES=0 python nllb_translator.py --noun [entity] --csv_path [csv-file containing the captions] --language [language of captions, e.g, spanish, hindi, greek, japanese]
```

## Distribution-based Analysis
To obtain the country and continent based distributions for an entity, run the following code, which returns two csvs containing the country and continent counts for the given entity respectively (including the counts for the special case for which the captions could not be geoprofiled to any country/continent).

```bash
python find_distribution.py --noun [entity] --tagged_file [csv-file containing the captions] --column [column containing the geoprofiled tags] --svm [Use if you want to find distribution for the relevant images] --dataset [laion-en, datacomp, cc12m] --language [for multilingual data, Jap/Spa/Gre/Hin] --svm_column [column name which stores if the image is relevant or irrelevant]
```

To get combine the distributions across all entities, run the following code, which merges the distributions across all entities, to return the overall frequency of the countries and the continents.

```bash
python find_distribution_combine.py --column [column name containing the names of the countries] --dataset [laion-en, datacomp, cc12] --language [for multilingual data, Jap/Spa/Gre/Hin]
```

## Diversity

To calculate how diverse the images are for a given noun and country,  
first install the vendi-score library.
```bash
pip install vendi_score
```
Go the location where the library is stored, and modify vendi_score/image_utils.py and vendi_score/text_utils.py to change the underlying feature encoder to CLIP ViTB/32. Then run the following code.
```bash
CUDA_VISIBLE_DEVICES=0 python image_diversity.py --noun [entity] --csv-path [path to the csv file containing the captions and image paths] --img_directory [image-directory] --dataset [laion-en, datacomp, Jap, Hin, Gre, Spa] --country-col [column name containing the geo-profiled country name] --annotation-col [column name containing the entity-presence classifier predictions] --vendi
```

Similarly for calculating diversity of captions, run the following code:
```bash
CUDA_VISIBLE_DEVICES=0 python text_diversity.py --noun [entity] --csv-path [path to the csv file containing the captions and image paths] --dataset [laion-en, datacomp, Jap, Hin, Gre, Spa] --country-col [column name containing the geo-profiled country name] --annotation-col [column name containing the entity-presence classifier predictions] --vendi
```