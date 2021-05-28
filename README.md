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
  * create a virtual enviourment using command :
     conda create --name myenv
    
  * install above mentioned python libraries.

#### To Host file locally #### 
  
  * Open python/anaconda terminal and use this command to deploy Flask  :
    
    Set FLASK_APP= "filename.py
    
    Set FLASK_DEBUG = 1
    
    flask run
    
   Now head over to the local host link shown in your terminal. Eg: *http://127.0.0.1:5000/*.
  
   For more detailed instructionon on Flask [click here](https://flask.palletsprojects.com/en/1.1.x/quickstart/).
 
 ## ***Shown Below are instructions on how to use the synthetic data generator*** ##

### 1. Module 1 ###    
   i)   Select total number of rows to be generated. this is essentially the amount of data that you want to be generated.
    
   ii)  Selct the type of file in which you would like to export the generated data.
    
   iii) Enter the name of the fields that you would like to generate the data for. 
           
   -  Ex. Name, Number, City etc.
   
   -  You can add next field by pressing the '+' button on the right side of field box.
    
   iv)  After this simply click the genrate button to genrate the data. 
          
### 2. Module 2 ###
   
   i)   Upload the dataset you would like your generated dataset to be based upon.
        (pls use clean data set)
   
   ii)  Select the number of rows of data you would like to generate.
   
   iii) Selct the type of file in which you would like to export the generated data.

### 3. Module 3 ###
   
   i)   Upload the data set you wish to apply machine learning algorithm on.
   
   ii)  Select the machinelearning algorithm you want to perform on the uploaded data set.
   
   iii) Input the value of X and Y from the data set.
   
   iv)  Click on generate button.   
   
