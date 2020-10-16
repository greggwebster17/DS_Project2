# Disaster Response Pipeline Project

### 1. Installations  

No additional libraries are needed to be installed in order for the project code to run.

### 2. Project Motivation  

The aim of this project is to classify messages into 36 different categories so that an appropriate reponse can be determined in certain disaster scnenarios. 
The project creates a database from distress messages which have been tagged by figure 8 into 36 categories and builds a classifier for each of these 36 categories. 
The database and classifier is then integrated into a web app which classifies any message into the 36 categories.

### 3. Instructions for running the code:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. In a new terminal run: env|grep WORK. The output of this will be a SPACEDOMAIN SPACEID- use these to fill the address https://SPACEID-3001.SPACEDOMAIN and put into a new webpage tab.

### 4. Acknowledgements

Credit to Figure 8 for supplying the data. There are the restrictions to the use of this code. 
