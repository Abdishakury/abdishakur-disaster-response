# **Disaster Response Pipeline Project**

- [Description](#Description)
- [Requirements](#Requirements)
- [Project Components](#Components)
- [File Descriptions](#File-Descriptions)
- [More Details](#Details)
- [Instructions](#How-To-Run-This-Project)
- [Screenshots](#Screenshots)
- [Licensing, Authors, Acknowledgements](#License)

## Description <a name="Description"></a>

The Disaster Response Pipeline Project is the data engineer project assigned by Udacity data scientist nanodegree. This project builds a **Machine Learning Natural Language Processing pipeline** to categorize emergency messages based on the needs communicated by the sender. 

### Website: https://abdishakur-disaster-response.herokuapp.com/

## Requirements <a name="Requirements"></a>

    click==8.0.3
    colorama==0.4.4
    Flask==2.0.2
    greenlet==1.1.2
    gunicorn==20.1.0
    itsdangerous==2.0.1
    Jinja2==3.0.3
    joblib==1.1.0
    MarkupSafe==2.0.1
    nltk==3.6.5
    numpy==1.22.4
    pandas==1.3.5
    plotly==5.4.0
    python-dateutil==2.8.2
    pytz==2021.3
    regex==2021.11.10
    scikit-learn==1.0.1
    scipy==1.7.3
    six==1.16.0
    SQLAlchemy==1.4.28
    tenacity==8.0.1
    threadpoolctl==3.0.0
    tqdm==4.62.3
    Werkzeug==2.0.2

## Project Components <a name="Components"></a>

1. **ETL Pipeline**
    1. Disaster_categories.csv and disater_messages.csv is imported as dataframe.
    2. Perform extract, transform and load to clean the dataset and save the dataset to SQL database.
2. **ML Pipeline**
    1. Load dataframe from database.
    2. Split the dataset into training and testing set.
    3. Train the model on training set and test the model on testing set, the model first tokenize, normalize, and lemmatize the dataset then apply Random Forest Classifier to do classification.
    4. Try grid search for best parameters and also try other model(such as Support Vector Machine, Adaboost, kNN, Decision Tree).
    5. Compare the models by looking at their precision, recall, and accuracy. Choose a model that have relative higher of these attributes.
    6. Export the model as a pickle file.
3. **Web Deployment**
    1. Create a html website and link to Github database
    2. Deploy app on Heroku so that everyone can view it.

## File Description <a name="File-Descriptions"></a>

```
├── app
│   ├── run.py--------------------------------# Flask file runs the web app
│   └── templates
│       ├── go.html---------------------------# Result page
│       └── master.html-----------------------# Main page
├── data
│   ├── DisasterResponse.db-------------------# *Database storing the processed data
│   ├── categories.csv---------------# Pre-labelled dataset
│   ├── messages.csv-----------------# Data
│   ├── process_data.py-----------------------# ETL pipeline processing the data
|   └── ETL Pipeline Preparation.ipynb-----# Jupyter notebook with details
├── models
|   ├── train_classifier.py-------------------# Machine learning pipeline
│   ├── ML Pipeline Preparation.ipynb------# Jupyter notebook with details
|   └── classifier.pkl------------------------# *Pickle file
*Files that will be generated when the python scripts .py are executed.
```

## More Details <a name="Details"></a>
It might also be a good idea to include more details about the project, such as how the the dataset formatted, the size of the dataset, how you cleaned the data, what models were tried etc.

The dataset `categories.csv` is a (26249 x 2) table that have columns: `id`,`categories`, while in `categories` column there are 36 categories that need to be split into 36 columns as output.

The dataset `messages.csv` is a (26249 x 4) table that have columns: `id`, `message`, `original`, `genre`, where `message` is used as an input.

The merged dataset is a table (26249 x 37) like: `message`,`categories1`,`categories2`,...,`categories36`.

The data cleaning steps:
1. Convert category values to just numbers 0 or 1.
2. Check if there are values other than 0 and 1 in categories.
3. If there are values other than 0 or 1 in `categories` columns, replace other values as 1
4. Drop duplicates.
5. Save to sql database.

The ML steps:
1. Clean text: remove punctuation, tokenize the text, remove stop words, Lemmatization.
2. Build nltk random forest pipeline.
3. Use GridSearchCV for best model (use low forest depth for try, since calculation takes hours).
4. save model to pickle file.

The web deployment steps:
1. Use web heroku to deploy app
2. Add Procfile, requirement.txt, runtime.txt, ntlk.txt

## Instructions <a name="How-To-Run-This-Project"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to the website that is showed in your command line, should be something like http://0.0.0.0:3001/ 

## Screenshots <a name="Screenshots"></a>
![Screen Shot](imgs/img1.png?raw=true)

## Licensing, Authors, Acknowledgements <a name="License"></a>
**Licensing**
Begin license text.
Copyright <2022> <COPYRIGHT Abdishakur Yoonis>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

End license text.

**Authors** [Abdishakur Yoonis](https://github.com/Abdishakury/)

**Acknowledgements** Abdishakur Yoonis,


