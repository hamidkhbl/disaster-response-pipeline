
## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
Steps to run the application:

1. Install the requirments:
>`pip install -r requirements.txt`

2. Run the Extract, Transform, Load (ETL) pipeline:<br> 
> `python ..\data\process_data.py` <br>
  This command will download csv files, merge and clean them. The clean data stores in a SQLlite database (`disaster_tweets.db`). If you don't want to download the files use this command:  <br>
`python process_data.py <message csv file> <category csv file> <database name>` <br>
You also can read all the logs from `process_data.log`.
3. Run ML and NLP pipeline: <br>
>`python ..\models\train_classifier.py` <br>
this command trains a Random Forest Classifier on the clean data and exports the model to a pickle file.
4. Run the Flask application <br>
> `python ..\app\run.py` <br>
Go to http://0.0.0.0:3001/

This code is tested on Python 3.7.6

## Project Motivation<a name="motivation"></a>
When a disaster such as a flood or a storm happens, people's tweet on the effected region contains useful information. In this project, we try to tag tweets based on the content of the tweet. This project can help to identify locations and people with a need to help.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to [Figure8](https://www.figure-eight.com/)  for the data.


