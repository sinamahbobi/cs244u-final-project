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

CS224U_Final_tfidf.ipynb is a Jupyter Notebook that is detailed with comments of each step and function for the TF-IDF recommendation system.

main_tfidf.py is the main scipt to run the TF-IDF Recommendation System in the Terminal/Command Line. 

CS224U_Final_PV-DM.ipynb is a Jupyter Notebook that is detailed with comments of each step and functionf for the PV-DM recommendation system.

main_pvdm.py is the main scipt to run the PV-DM Recommendation System in the Terminal/Command Line. 

input_lyrics1.txt (Sam Smith's Too Good at Goodbye song) is an input lyric can be used to test out the both recommendation system.

output.json is the expected output of the TF-IDF RS after using the provided input in the main script.

--------------------------------------------------------------------------------------

TO RUN THE MAIN SCRIPT:

1. Make sure all the above files, including the data file (final_lyrics.csv) are in the same directory/folder (output.json is not required).
2. Using the Terminal, make sure you are in the correct folder (i.e. cd /Users/Name/xcs244u)
3. Run this command: python3 main_tfidf.py final_lyrics.csv input_lyrics1.txt WhateverYouWantToNameYourOutputFile.json
-  main_tfidf.py is also interchangable with main_pvdm.py
5. If running the TF-IDF RS, this should output the same recommended song as in the output.json file. The PV-DM outputs a different recommended song.
6. To try other input lyrics, simply create txt files of just the lyrics and use that instead of input_lyrics1.txt in the command.
