# online-sentiment-analysis

This project uses Natural Language Processing (ML) to detect the sentiment of certain keywords/terms on Social Media (currently implemented for BlueSky, but can be modified for many websites). This program is trained on a dataset of 1.6 million tweets with positive (1) & negative (0) sentiment ratings, and has a 79.8% accuracy rate when tested against its own data.

When running this program, you will be asked for your BlueSky login and a keyword/term you would like to search for (e.g. sixers, nosferatu, new york). The program will then pull 100 of the top posts under the provided keyword, and test the posts against the trained sentiment model to detect the frequency of positive vs negative sentiment.

The program is functional, but the user interface is still under development. 

