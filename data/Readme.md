GeoNames test set: captions_population_10K_1.csv (caption in col 'text', ground truth location in col 'new_place', ground truth country in col 'new_gt')
Wikipedia test set: marginalized_countries_1.csv (caption in col 'Text', ground truth location in col 'Place Name', ground truth country in col 'Country name EN')
LAION self annotated subset: laion_selfannotation.csv (caption in col 'text', no ground truth location annotated, ground truth country in col 'gt')

Each csv has a column called 'train'. 'train'==0 refers to the train set, 'train'==1 refers to the val set, and 'train'==2 refers to the test set.

As minimal training is involved in tuning the prompts, we have kept a 10:10:80 split for train, val and test.
