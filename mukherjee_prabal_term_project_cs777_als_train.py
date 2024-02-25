"""
Name: Prabal Mukherjee
Class: MET CS 777 - Big Data Analytics - Spring 1 2024
Date:  02-23-2024
Assignment: Homework - Term Project
Description: 
       Script to train ALS model
       1. Performace tuning
       2. Saving recommendationoutput for use 
Professor: Prof. Dimitar Trajanov, PhD.
Facilitar: Nathan A. Horak (Group 2)
"""


from __future__ import print_function

import sys
from sys import exit
import traceback
import time
from pyspark import SparkContext
from pyspark.sql import SparkSession

import os
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

if __name__ == "__main__":
    print("Input arguments are : ",sys.argv)
    if len(sys.argv) != 3:
        print("Usage: main_task1 <file> <output> ", file=sys.stderr)
        exit(-1)
    
    sc = SparkContext(appName="Pmukhe11-CS777-Term-Project")
    spark = SparkSession.builder.getOrCreate()
    #sc.addFile(r'gs://pmukhe11_cs777_spring24_lab_bucket/TermProject/mukherjee_prabal_term_project_cs777_helper.py')
    sc.addFile(r'C:/Users/praba/OneDrive/BU-MSCIS/CS 777 Big Data Analytics/Term Project/cs-777-termproject-PrabalMukherjee/mukherjee_prabal_term_project_cs777_helper.py')
    import mukherjee_prabal_term_project_cs777_helper as helper

    print('Added helper file')
    
    try:
        start_time = time.perf_counter()
        reviewfilename = sys.argv[1]
        outputFolder = sys.argv[2]

        print('\nLoading - Pre Processing Review dataframe')
        dfReview = helper.loadReviewDF(spark, reviewfilename)
        #dfReview = dfReview.sample(withReplacement=False, fraction=0.02, seed=3)

        dfReview.cache()
        helper.trainALSModel(sc, dfReview, outputFolder)

        #Save a trained recommendation
        helper.getrecomendation(dfReview,"dummy", True, outputFolder)

        print('\nCommand completed successfully')
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time in secs: {elapsed_time}")

    except Exception as e:
        print(e)
        print(traceback.format_exc())
    finally:
        sc.stop()


    
