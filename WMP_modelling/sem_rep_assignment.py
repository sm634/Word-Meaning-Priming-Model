from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
from Nltk.wordnet_feature_extraction_functions import get_features

with open("""C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access Model\\semantic representation\\noun_homographs_final.csv""") as f:
    words_list = f.readlines()
    words_list = [word.replace(',','').replace('\n','') for word in words_list]
f.close()

with open("""C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access Model\\semantic representation\\features.csv""") as f:
    features_list = f.readlines()
    features_list = [feature.replace(',','').replace('\n','') for feature in features_list]
f.close()


def assign_feature(word, n_synsets, feature_list: list) -> np.array:
    feature_array = np.zeros([len(feature_list), 2])
    synset_features = get_features(word, n_synsets)
    for i, synset in enumerate(synset_features):
        for features in synset.values():
            for feature in features:
                feature_array[feature_list.index(feature), i] = 1
    return feature_array

df_columns = []
for word in words_list:
    for i in range(0,2):
        df_columns.append(wn.synsets(word)[i].name())


semantic_representations = np.array([len(features_list), 2])

for i, word in enumerate(words_list):
    if i == 0:
        semantic_representations = assign_feature(word, 2, features_list)
    else:
        semantic_representations = np.concatenate((semantic_representations,
                                                  assign_feature(word, 2, features_list)), axis=1)

sem_rep_df = pd.DataFrame(data=semantic_representations,columns=df_columns)
sem_rep_df.to_csv('C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access Model\\semantic representation\\semantic_representations.csv')

headers = ['Word','Synset 1','Synset 2']
words_synsets = open("""C:\\Users\\sm634\\OneDrive\\Desktop\\Folder\\Research\\Word Access Model\\semantic representation\\words_synsets.csv""",'w')

for i, header in enumerate(headers):
    words_synsets.write(header+',')

for i, word in enumerate(words_list):
    row = '\n{},{},{}'.format(word, wn.synsets(word)[0].name(), wn.synsets(word)[1].name())
    words_synsets.write(row)

words_synsets.close()






