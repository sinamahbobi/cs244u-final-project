### Requirements
# For text processing
# - pip install gensim
# - pip install pyldavis

# For data cleaning - removing any language other than English
# - pip install pycld2

# To test for encoding in dataset
# - pip install chardet

# import required packages/libraries
import sys
import pandas as pd
import json
from operator import itemgetter
# import chardet (used to chekc encoding of datafile)
import pycld2 as cld2
# import custom filters
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric, stem_text
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short
# for tfidf and cosine sim calculation
from gensim import corpora, models, similarities


def create_df(data_filename):
    lyrics_data = pd.read_csv(data_filename, encoding='utf-8')
    # drop first col
    lyrics_df = lyrics_data.iloc[: , 1:]
    return lyrics_df

def data_cleaning(lyrics_df):
    # fix encoding and remove special characters and remove irrelevant words
    lyrics_df['Lyrics'] =  [str(lyrics.encode('ascii', 'replace')).replace('b"','').replace('?',' ').replace('"','').replace('\\n', ' ').replace("b'",'').replace('instrumental','').replace('[\[],:*!?]','').replace('(','').replace(')','').replace('.','').replace(',','').replace('\\','').replace('verse','').replace('!','').replace('chorus','').replace('*','')
               for lyrics in lyrics_df['Lyrics'].str.decode('unicode_escape')]
    
    # filter for only English lyrics
    en_lyrics = []
    for i in range(len(lyrics_df)):
        _, _, _, detected_language = cld2.detect(lyrics_df['Lyrics'][i],  returnVectors=True)
        if len(detected_language) == 1:
            if detected_language[0][2] == 'ENGLISH':
                en_lyrics.append(i)

    lyrics_df = lyrics_df.iloc[en_lyrics,:] 
    # data cut from 243406 to 211873

    # fix df index
    lyrics_df.reset_index(inplace=True)
    return lyrics_df

def preprocessing(lyrics_df, CUSTOM_FILTERS):
    text_preprocess = []
    for i in range(len(lyrics_df)):
        text_preprocess.append(preprocess_string(lyrics_df['Lyrics'][i], CUSTOM_FILTERS))
    lyrics_df['text_preprocessing'] = text_preprocess
    return lyrics_df

def add_topic_to_df(lyrics_df, CUSTOM_FILTERS, lyrics_dictionary, lda):
    topic_list = []
    for i in range(len(lyrics_df)):
        pp_lyrics = preprocess_string(lyrics_df['Lyrics'][i], CUSTOM_FILTERS)
        bow_lyrics = lyrics_dictionary.doc2bow(pp_lyrics)
        topics = lda.get_document_topics(bow_lyrics)
        topic = max(topics, key=itemgetter(1))[0]
        topic_list.append(topic)
        
    lyrics_df['topic'] = topic_list
    return lyrics_df

def input_data(input_filename, CUSTOM_FILTERS):
    with open(input_filename, 'r') as f:
        input_lyrics = f.read()
        # clean input lyrics
        input_lyrics = input_lyrics.replace('b"','').replace('?',' ').replace('"','').replace('\\n', ' ').replace("b'",'').replace('instrumental','').replace('[\[],:*!?]','').replace('(','').replace(')','').replace('.','').replace(',','').replace('\\','').replace('verse','').replace('!','').replace('chorus','').replace('*','')
        # pre-process
        input_lyrics_preprocess = preprocess_string(input_lyrics, CUSTOM_FILTERS)
        return input_lyrics_preprocess


if __name__ == '__main__':

    # command inputs
    data_filename = sys.argv[1]
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]

    # define custom filters
    CUSTOM_FILTERS = [lambda x: x.lower(), #lowercase
                    strip_multiple_whitespaces,# remove repeating whitespaces
                    strip_numeric, # remove numbers
                    remove_stopwords,# remove stopwords
                    strip_short # remove words less than minsize=3 characters long
    #                   stem_text # return porter-stemmed text,
                    ]

    #get cleaned and pre-processed df
    # lyrics_df = create_df(data_filename)
    # lyrics_df = data_cleaning(lyrics_df)
    lyrics_df = preprocessing(data_cleaning(create_df(data_filename)), CUSTOM_FILTERS)
    #create dictionary
    lyrics_dictionary = corpora.Dictionary(lyrics_df['text_preprocessing'])
    # convert tokenized documents to vectors
    corpus = [lyrics_dictionary.doc2bow(text) for text in lyrics_df['text_preprocessing']]

    #TF-IDF and LDA
    # initialize tfidf model
    tfidf = models.TfidfModel(corpus)
    # apply transformation to entire corpus
    transformed_tfidf = tfidf[corpus]
    # LDA on tfidf
    lda = models.LdaMulticore(transformed_tfidf, num_topics=4, id2word=lyrics_dictionary)
    # add topic to df
    lyrics_df = add_topic_to_df(lyrics_df, CUSTOM_FILTERS, lyrics_dictionary, lda)

    #Similarity Calculation
    # INPUT
    input_lyrics = input_data(input_filename, CUSTOM_FILTERS)
    # TF-IDF - vectorize lyrics
    bow_input_lyrics = lyrics_dictionary.doc2bow(input_lyrics)
    # LDA - get topic
    input_lyrics_topics = lda.get_document_topics(bow_input_lyrics)
    input_lyrics_lda = max(input_lyrics_topics, key=itemgetter(1))[0] 
    # apply transformation to input lyrics
    transform_input_lyrics = tfidf[bow_input_lyrics]
    # get lyrics matrix
    index = similarities.SparseMatrixSimilarity(transformed_tfidf, num_features=len(lyrics_dictionary))
    # calc cosine similarity of input with every item in the dataset
    cos_sims = index[transform_input_lyrics]
    # create new df based on matched topic
    topic_df = lyrics_df[lyrics_df.topic == input_lyrics_lda]
    # filter for matching topic
    topic_ind = lyrics_df[lyrics_df.topic == input_lyrics_lda].index.tolist()
    topic_song_list = [cos_sims[i] for i in topic_ind]

    # get index of reccommended song
    song_index = topic_song_list.index(max(topic_song_list))
    rec_song = {'Artist': topic_df['Artist'][song_index],
                    'Title': topic_df['Title'][song_index],
                    'Genre': topic_df['Genre'][song_index],
                    'Lyrics': topic_df['Lyrics'][song_index]}

    with open(output_filename, 'w') as fout:
        json.dump(rec_song, fout)
