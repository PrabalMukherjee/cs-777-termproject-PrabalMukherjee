
# Name: Prabal Mukherjee
# Class: MET CS 777 - Big Data Analytics - Spring 1 2024
# Date:  02-16-2024
# Assignment: Homework - - Term Project
# Description: 

       
       How can natural language processing techniques be leveraged to develop an efficient search 
       algorithm for product discovery on the Amazon Review Dataset? Additionally, 
       how can the top reviews within a specific product category be utilized to create a recommender system, 
       ultimately presenting users with a curated list of top 10 products for purchase?
       
# Professor: Prof. Dimitar Trajanov, PhD.
# Facilitar: Nathan A. Horak (Group 2)

# Folders:

> .\data\input   : contains input data files
Taken from https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Digital_Music_5.json.gz

> .\output_*     : contains spark output

# Files

> mukherjee_prabal_term_project_cs777_extract_data.py : </b>
  Script that takes input dataset in gzipped format </b>
  and extract into json format

> mukherjee_prabal_term_project_cs777_als_train.py > 
  Script for Performace Tuning and training ALS models </b>

> mukherjee_prabal_term_project_cs777_task.py > </b>
  Main task script to get search result in TFIDF and other techniques based on </b>
  parameters, and get recommendation from ALS model

> mukherjee_prabal_term_project_cs777_helper.py> </b>
  Helper module that contains all the function like loading, claening preprocessing dataset <b>
  Creating pipelines for different transformation, get top 10 records using cosine calculator
  training and tuning ALS model and getting recommendation.

