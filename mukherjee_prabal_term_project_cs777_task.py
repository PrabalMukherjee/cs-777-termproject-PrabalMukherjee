"""
Name: Prabal Mukherjee
Class: MET CS 777 - Big Data Analytics - Spring 1 2024
Date:  02-23-2024
Assignment: Homework - Term Project
Description: 
       How can natural language processing techniques be leveraged to develop an efficient search 
       algorithm for product discovery on the Amazon Review Dataset? Additionally, 
       how can the top reviews within a specific product category be utilized to create a recommender system, 
       ultimately presenting users with a curated list of top 10 products for purchase?

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
import sparknlp


import os
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

if __name__ == "__main__":
    print("Input arguments are : ",sys.argv)
    if len(sys.argv) != 5:
        print("Usage: main_task1 <file> <output> ", file=sys.stderr)
        exit(-1)
    
    sc = SparkContext(appName="Pmukhe11-CS777-Term-Project")
    spark = SparkSession.builder.getOrCreate()
    #This code is not tested in gcloud, if you manage to run cluster with sparknlp you can try, if you uncoment line 41
    #sc.addFile(r'gs://pmukhe11_cs777_spring24_lab_bucket/TermProject/mukherjee_prabal_term_project_cs777_helper.py')
    sc.addFile(r'C:/Users/praba/OneDrive/BU-MSCIS/CS 777 Big Data Analytics/Term Project/cs-777-termproject-PrabalMukherjee/mukherjee_prabal_term_project_cs777_helper.py')
    import mukherjee_prabal_term_project_cs777_helper as helper

    print('Added helper file')
    
    try:
        start_time = time.perf_counter()
        reviewfilename = sys.argv[1]
        question = sys.argv[2]
        outputFolder = sys.argv[3]
        method = sys.argv[4]
        numTopWords = 5000

        print('\nLoading - Pre Processing Review dataframe')
        dfReview = helper.loadReviewDF(spark, reviewfilename)
        dfReview = dfReview.sample(withReplacement=False, fraction=0.02, seed=3)

        if(method == 'TFIDF'):
            print('\Generating top 10 review using TFIDF')
            df = helper.getTop10ReviewByTFIDF(numTopWords, dfReview, question, outputFolder)
        elif(method == 'WORD2VEC'):
            print('\Generating top 10 review using WORD 2 vec')
            df = helper.getTop10ReviewByWord2Vec(dfReview, question, outputFolder)
        elif(method == 'BERT'):
            #spark = sparknlp.start()
            print('\Generating top 10 review using BERT')
            helper.getTop10ReviewByBERT(dfReview, question, outputFolder)

        userId = df.first()['reviewerID']
        df_rec = helper.getrecomendation(dfReview, userId)
        df_rec.show()

        print('\nCommand completed successfully')
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time in secs: {elapsed_time}")

    except Exception as e:
        print(e)
        print(traceback.format_exc())
    finally:
        sc.stop()


    
