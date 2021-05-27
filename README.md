# SynGen #
----
#### A Simple Approach To Synthetic Data Generation #####

This repository contain all the files needed to run the synthetic data generator.


### Purpose : ###
The goal of this project is to develope a synthetic data generator which can provide user the best 
experience with minimal user interactions while generating quality synthetic data all the while 
preserving similar patterns and a similar statistical distribution of a real dataset.

Few significant features of the developed platform are:
1.  To provide a platform for synthetic data generation based on user defined schema.
2.  To generate the synthetic data based on the data which is already available with the user.
3.  To provide a framework wherein users can upload their own dataset and can apply various machine learning algorithms and compare the results.

### Requirements : ###
Creating a virtual environment is strongly suggested. [Click here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html ) to know more about
virtual enviroments using Anaconda.

Following python packages would be required for running the project:
  * matplotlib
  * numpy
  * Faker
  * Flask
  * Flask_MySQLdb
  * seaborn
  * Werkzeug
  * pandas
  * PyYAML
  * scikit_learn
  
  ### Usage ###
  1. create a virtual enviourment using command :
  conda create --name myenv
    
  2. install above mentioned python libraries.
  
  3. Open python/anaconda terminal and use this command to deploy Flask  :
    
    Set FLASK_APP= "filename.py
    
    Set FLASK_DEBUG = 1
    
    flask run
    
   Now head over to the local host link shown in your terminal. Eg: *http://127.0.0.1:5000/*.
  
   For more detailed instructionon Flask [click here](https://flask.palletsprojects.com/en/1.1.x/quickstart/).
 
  
