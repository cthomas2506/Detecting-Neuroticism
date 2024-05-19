1.	Code setup:

            i.	Clone the GitHub Repository
            ii.	Use an IDE, preferably IntelliJ Pycharm
            iii.	Setup a python interpreter. We used python version 3.10
            iv.	Setup a virtual environment, preferably a conda environment.
            v.	Once the environment is setup, go inside the ‘DATA641 Final Project’ directory. 
            vi.	Use the command ‘pip install -r requirements.txt’ to install the required modules
            vii.	Please note that some modules might not get installed. One can use conda to install them
            viii.	Once all the modules have been installed, step 1 of code setup is done!


2.	Placing the necessary files:

          i.	A few files that are too large to be placed as a part of the github repo must be manually placed.
          ii.	The first set of files are the GloVe word embeddings. 
          iii.	Go to their website: https://nlp.stanford.edu/projects/glove/
          iv.	Download the glove.6B.zip file and unzip it to your system.
          v.	Place the ‘glove.6B.50d.txt’ and ‘glove.6B.2000d.txt’ files inside the project in this path: DATA641 Final Project ->  resources
          vi.	Please also make sure that the essay CSV file renamed as ‘wcpr_essays.csv’ and the personality CSV file renamed as ‘wcpr_personality.csv’ are present in the DATA641 Final Project -> data -> csvs folder.
          vii.	Once these checks have been completed, we’re ready to get started!


3.	There are a bunch of programs that can be run, the important ones are listed below:

a.	Hyperparameter tuned traditional ML Models

      i.	To run this, go to ‘DATA641 Final Project -> codes -> driver -> main.py’
      ii.	You can play around, comment and uncomment various sections of the code in the __main__ function and fetch results as required. 

b.	CNN Model

    i.	To run the CNN model, go to ‘DATA641 Fianl Project -> codes -> models -> cnn.py’
    ii.	All you have to do is run the __main__ function.
    iii.	Whilst all the variables are setup, one can play around with the parameters of the model. 

c.	Ensemble Learning Model 

    i.	To run the ensemble learning model, go to ‘DATA641 Final Project -> codes -> classifiers -> ensemble_learning.py’
    ii.	There’s not much to play around here. Just run the __main__ function. Do not change any of the codes. The only changeable part is the dataset and word embeddings dimensions. 

d.	Majority tagging and random tagging models

    i.	These are the two most basic and simple models. To run them, go to ‘DATA641 Final Project -> driver -> tagging.py’
    ii.	You can play around with the variables 
    iii.	Once the parameters are set to your liking, run the __main__ function and you shall see the results.
