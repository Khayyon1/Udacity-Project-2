# README Criteria
The README file includes a summary of the project, how to run the Python scripts and web app, and an explanation of the files in the repository. Comments are used effectively and each function has a docstring.

# The Disaster Response Pipeline Project
This project is a combination of a ETL pipeline process that cleans the datasets provided and stores the cleaned data into a Database file to be used by both the machine learning pipeline to create the model that is used by the web application to classify messages provided by the users in the browser to the appropriate class of disaster categories. 

## Project Structure
* [app](./app) - this directory contains the flask web application for this project
    * [templates](./app/templates) - this directory contains the html files for the web app
        * [go.html](./app/templates/go.html) - The result page of the web app that displays at least 2 visualizations and the results from the ML model
        * [master.html](./app/templates/master.html) -  The home page of the web app 
    * [custom_extractor.py](./app/custom_extractor.py)
    * [run.py](./app/run.py) - the flask application that runs on https://localhost:3000. It automatically uses the ML model from the ML pipeline
* data
    * [disaster_categories.csv](./data/disaster_categories.csv) - dataset containing the different genres for each of the messages
    * [disaster_messages.csv](./data/disaster_messages.csv) - dataset contains the disaster messages to be classified by the model
    * [DisasterResponse.db](./data/DisasterResponse.db) - The database file created from the ETL Pipeline that stored the cleaned data from the two datasets required for the ML Pipeline and the Web Application.
    * [process_data.py](./data/process_data.py) - This is the ETL pipeline script that generates the Database needed for the rest of the project. This combines the data from the two datasets, and cleans it up before creating or storing the data into the Database (i.e. based on if the DB exist or not). 
* models
    * [classifier.pkl](./models/classifier.pkl) - Machine Learning model generated from the train_classifier.py script. Used by the web app to return results to the go page 
    * [custom_extractor.py](./models/custom_extractor.py) -
    * [train_classifier.py](./models/train_classifier.py) - The ML Pipeline that uses the Database generated from the ETL pipeline to train the model and useds Grid Search to optimize it as well. It outputs the model as a classifier.pkl file upon successful execution 
    

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Technologies Used
- Python 3.10
- scikit-learn
- Numpy
- Pandas
- SQlchemy
- flask
- pickle
- HTML
- plotly
