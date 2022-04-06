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
# from operator import itemgetter
# import chardet (used to chekc encoding of datafile)
import pycld2 as cld2
# for pv-dm and cosine sim calculation
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


def create_df(data_filename):
    lyrics_data = pd.read_csv(data_filename, encoding='utf-8')
    # drop first col
    lyrics_df = lyrics_data.iloc[: , 1:]
    return lyrics_df

def data_cleaning(lyrics_df):
    # fix encoding and remove special characters and remove irrelevant words
    lyrics_df['Lyrics'] =  [str(lyrics.encode('ascii', 'replace')).replace('b"','').replace('?',' ').replace('"','').replace('\\n', ' ').replace("b'",'').replace('instrumental','').replace('[\[],:*!?]','').replace('(','').replace(')','').replace('.','').replace(',','').replace('\\','').replace('verse','').replace('!','').replace('chorus','').replace('*','').replace('\n',' ')
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

def input_data(input_filename):
    with open(input_filename, 'r') as f:
        input_lyrics = f.read()
        # clean input lyrics
        input_lyrics = input_lyrics.replace('b"','').replace('?',' ').replace('"','').replace('\\n', ' ').replace("b'",'').replace('instrumental','').replace('[\[],:*!?]','').replace('(','').replace(')','').replace('.','').replace(',','').replace('\\','').replace('verse','').replace('!','').replace('chorus','').replace('*','').replace('\n',' ')
        return input_lyrics


if __name__ == '__main__':

    # command inputs
    data_filename = sys.argv[1]
    input_filename = sys.argv[2]
    output_filename = sys.argv[3]

    #get cleaned df (no need preprocessing)
    lyrics_df = data_cleaning(create_df(data_filename))

    #create corpus
    #Use cleaned complete lyrics instead of pre-processed text
    corpus = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(lyrics_df['Lyrics'])]
    
    #build model
    #100 epochs is impractically long
    # higher vec_size value increases performace but also takes longer/memory
    # max_epochs = 100
    vec_size = 100
    alpha = 0.025
    # dm=1 is for PV-DM
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.0025,
                    min_count=1,
                    dm =1,
                    epochs=10)

    model.build_vocab(corpus)

    # train model
    model.train(corpus,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    # # Train model with Hyper-paramenter tuning --> takes long as well!!
    # for epoch in range(max_epochs):
    #     print('iteration {0}'.format(epoch))
    #     model.train(corpus,
    #                 total_examples=model.corpus_count,
    #                 epochs=model.epochs)
    #     # decrease the learning rate
    #     model.alpha -= 0.0002
    #     # fix the learning rate, no decay
    #     model.min_alpha = model.alpha
        
    #save model
    model.save("pvdm_model")

    #load model
    model= Doc2Vec.load("pvdm_model")

    #Similarity Calculation
    # INPUT
    input_lyrics = input_data(input_filename)
    #tokenization - get doc vector
    input_data = word_tokenize(input_lyrics.lower())
    lyric_vector = model.infer_vector(input_data)
    # to find most similar songs
    similar_doc = model.docvecs.most_similar(lyric_vector)
    # # get index of reccommended song (closest/simialr song)
    song_index = int(similar_doc[0][0])

    # OUTPUT reccommended song
    rec_song = {'Artist': lyrics_df['Artist'][song_index],
                    'Title': lyrics_df['Title'][song_index],
                    'Genre': lyrics_df['Genre'][song_index],
                    'Lyrics': lyrics_df['Lyrics'][song_index]}

    with open(output_filename, 'w') as fout:
        json.dump(rec_song, fout)
