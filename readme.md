# esportsLABgg Counter-Strike Data Challenge
## Table of contents
* [General info](#general-info)
* [Files description](#files-description) 
* [Data description](#data-description)
* [How to run scripts](#how-to-run-scripts)

## General info 
This is my solution for esportsLABgg Counter-Strike Data Challenge. I had to create a python binary classifier for labeling grenade throws as correct/incorrect. My solution had to be a single python script named "classify.py" which reads from the command-line input the name of a file with grenade features. Script should overwrite test dataset with new column called "LABEL". 
## Files description
I will briefly explain what some files are for. I will not describe all the files.
The "scripts" folder contains the code files. There are :  
* main.ipynb - this is file with all work with data.
* classify.py - file which predict labels to given data 
* data_separation.py - script to separate dataset to train and test. Also it concat data from de_mirage and de_inferno maps. 
* model_eval.py - model evaluation
* best_model.pkl - model serialized parameters

Rest of files :  
* Conuter_Stike_solution_explanation.pdf - this is file where I described my thinking process.
* eSportsLab_Counter_Challange.pdf - in this file the competition is described in more detail

## Data description

Each grenade throw recorded in the input CSV file (single row) is described usingthe following set of features (the CSV contains also set of id values for company interial use:  demoid, demoroundid,roundstarttick, weaponfireid)  
* team:T– terrorists,CT– counter-terrorists
* (detonation_raw_x,detonation_raw_y,detonation_raw_z):  grenade detonation raw coordinates
* (throw_from_raw_x,throw_from_raw_y,throw_from_raw_z):  raw coordinates of the player when the grenade isbeing thrown
* throw_tick:  the exact tick (unit of game time, 128 ticks per second, counted from the beginning of the game),when the grenade is being thrown
* detonation_tick:  the exact tick, when the grenade is being detonated
* TYPE: type of the grenade (smoke,flashbang,molotov)
* map_name:  map on which the match was played (de_inferno, de_mirage)

Data for the competition included training datasets consisting of grenade throw data from two major competitive maps: de_inferno and de_mirage, each grenade throw is labeled using a boolean value. The meaning of the values in column ’LABEL’ is either correct throw (TRUE) or incorrect throw (FALSE).  
There are two sets of labeled training data:
* train-grenades-de_inferno.csv:  features of 354 grenade throws on de_inferno map
* train-grenades-de_mirage.csv:  features of 370 grenade throws on de_mirage map

# How to run scripts
First of all download the necessary packages from requirements.txt  
``` 
pip install -r requirements.txt
```
Next go to scripts folder.  
Separate the datasets with data_separation.py file. In my repository datasets are already separated so this step can be skipped.  
```
python data_separation.py
```
Make new predictions with classify.py file.  
```
python classify.py test.csv
```
test.csv is file with data without labels.  
Now evaluate model. As true labels I use test_labels.csv file which was created earlier with data_separation.py script.  
```
python model_eval.py
```