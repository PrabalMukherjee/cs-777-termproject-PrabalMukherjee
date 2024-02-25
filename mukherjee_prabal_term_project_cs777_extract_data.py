"""
Name: Prabal Mukherjee
Class: MET CS 777 - Big Data Analytics - Spring 1 2024
Date:  02-25-2024
Assignment: Homework - Term Project
Description: 
       Script to extract input data from gz fromat to json
Professor: Prof. Dimitar Trajanov, PhD.
Facilitar: Nathan A. Horak (Group 2)
"""


from __future__ import print_function

import sys
from sys import exit
import traceback
import time
import gzip
import json

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def extractReview(path, outputpath):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  listJson = [v for k,v in df.items()]
  with open(outputpath, 'w') as file:
        print('\n Dumping to ouputpath.')
        json.dump(listJson, file, indent=4)

  return df

if __name__ == "__main__":
    print("Input arguments are : ",sys.argv)
    if len(sys.argv) != 2:
        print("Usage: main_task1 <file> <output> ", file=sys.stderr)
        exit(-1)
    
    
    try:
        start_time = time.perf_counter()
        inputfile = sys.argv[1]
        outputpath = inputfile.replace('.json.gz','_c.json')
        print(f'\nInput file: {inputfile}')
        print(f'\nOutput file: {outputpath}')
        extractReview(inputfile, outputpath)
        print('\nCommand completed successfully')
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time in secs: {elapsed_time}")

    except Exception as e:
        print(e)
        print(traceback.format_exc())
    finally:
        print('\nexit finally')


    
