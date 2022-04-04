# cs244u-final-project

### Requirements
For text processing
- pip install gensim
- pip install pyldavis

For data cleaning - removing any language other than English
- pip install pycld2

To test for encoding in dataset
- pip install chardet

---------------------------------------------------------------------------------------

CS224U_Final_tfidf.ipynb is a Jupyter Notebook that is detailed with comments of each step and function.

main_tfidf.py is the main scipt to run the Recommendation System in the Terminal/Command Line. 

input_lyrics1.txt (Sam Smith's Too Good at Goodbye song) is an input lyric to test out the TF-IDF recommendation system.

output.json is the expected output after using the provided input in the main script.

--------------------------------------------------------------------------------------

TO RUN THE MAIN SCRIPT:

1. Make sure all the above files, including the data file (final_lyrics.csv) are in the same directory/folder (output.json is not required).
2. Using the Terminal, make sure you are in the correct folder (i.e. cd /Users/Name/xcs244u)
3. Run this command: python3 main_tfidf.py final_lyrics.csv input_lyrics1.txt WhateverYouWantToNameYourOutputFile.json
4. This should output the same recommended song as in the output.json file.
5. To try other input lyrics, simply create txt files of just the lyrics and use that instead of input_lyrics1.txt in the command.
