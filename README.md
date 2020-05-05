
## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Steps to run the application:
1. Run the Extract, Transform, Load (ETL) pipeline:<br> `python ..\data\process_data.py` <br>
  This command will download csv files, merge and clean them. The clean data stores in a SQLlite database (`disaster_tweets.db`). If you don't want to download the files use this command:  <br>
`python process_data.py <message csv file> <category csv file> <database name>` <br>
You also can read all the logs from `process_data.log`.
2. Run ML and NLP pipeline: <br>
`python ..\models\train_classifier.py` <br>
this command will train a Random Forest Classifier on the clean data and export the model to a pickle file.
3. Run the Flask application <br>
`python ..\app\run.py`

## Project Motivation<a name="motivation"></a>
When a disaster such as flood or storm happens, people's tweet on the effected region contains usefull information. In this projec we try to tag tweets based on the content of the tweet. This can help identofying locations and people with a need to help.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to [Figure8](https://www.figure-eight.com/)  for the data.


