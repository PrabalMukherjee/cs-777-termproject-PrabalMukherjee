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

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import lit
from pyspark.sql import functions as F
from pyspark.ml.feature import IDF, Tokenizer, CountVectorizer, StopWordsRemover, Word2Vec

from pyspark.ml import Pipeline
import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

from sparknlp.base import DocumentAssembler, EmbeddingsFinisher
from sparknlp.annotator import SentenceDetector, BertSentenceEmbeddings
from sparknlp.pretrained import PretrainedPipeline
from scipy.spatial import distance
from pyspark.sql.window import Window


dot_udf = F.udf(lambda x, y: float(distance.cosine(x,y)), DoubleType())
dot_udf_bert = F.udf(lambda x, y: float(distance.cosine(np.array(x[0]),np.array(y[0]))), DoubleType())

def null_check(str_val):
    return (len(str_val.strip())==0)

# check if the value is float or not
def isfloat(value):
    try:
        float(value)
        return True
    except:
         return False
     
def cleanupRatingRow(p):
    if(len(p) == 4):
        if((null_check(p[0]) or null_check(p[1]) or null_check(p[2]) or null_check(p[3])) == False):
            if(isfloat(p[3])):
                return p


#Loading rating dataframe
def loadRatingsinDF(sc, spark, fileRating):
    ratingRDD = sc.textFile(fileRating)
    ratingRDD = ratingRDD.map(lambda x: (x.split(','))).filter(lambda x: cleanupRatingRow(x)).map(lambda x: (x[0],x[1],float(x[2]),int(x[3])))
    ratingDF = spark.createDataFrame(ratingRDD, ['item','user','rating','timestamp'])
    return ratingDF

#loading review dataframe
def loadReviewDF(spark, fileReview):
    df = spark.read.option("multiline","true").json(fileReview)
    #Step 1. Project asin,overall,reviewText, reviewerID , summary from the datset 
    df_review = df.select(["asin","overall","reviewText","reviewerID","summary"])
    #Step2: Concatenate summary and review and build one reviewText
    df_review = df_review.withColumn('review', concat(col('summary'), lit(' ') , col('reviewText')))
    # Remove words 1, 2 char
    df_review = df_review.withColumn("text", F.regexp_replace("review", r"\b\w{1,2}\b", "")).cache()
    # Remove NA
    df_review = df_review.dropna(subset=["text"])
    return df_review

#create tf-idf pipeline
def createTFIDFPipeline(numTopWords):
    #Step 3: Create pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    # Remove stop words
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
    # Create a count vectorizer
    countVectorizer = CountVectorizer(inputCol=remover.getOutputCol(), outputCol="rawFeatures", vocabSize=numTopWords)
    # Calculate the TF-IDF
    idf = IDF(inputCol=countVectorizer.getOutputCol(), outputCol="features")
    # Crate a preprocessing pipeline with 4 stages
    pipeline_p = Pipeline(stages=[tokenizer,remover, countVectorizer, idf])
    return pipeline_p

#find top 10 review 
def getTop10Review(data_model, transformed_data, questionText, isBert= False):
    transformed_data = transformed_data.select("asin","overall","reviewerID","text","features")
    transformed_data = transformed_data.withColumnRenamed("text","reviewText").withColumnRenamed("features","reviewText_features")
    transformed_data = transformed_data.select("asin","overall","reviewerID","reviewText","reviewText_features")
    #transformed_data.cache()
    transformed_data = transformed_data.withColumn("text",lit(questionText))
    transformed_data = data_model.transform(transformed_data)
    transformed_data = transformed_data.select("asin","overall","reviewerID","reviewText","reviewText_features","features").withColumnRenamed("features","question_features")
    
    if(isBert):
        transformed_data = transformed_data.withColumn('cosine_similarity', dot_udf_bert('reviewText_features', 'question_features'))
    else:
        transformed_data = transformed_data.withColumn('cosine_similarity', dot_udf('reviewText_features', 'question_features'))

    transformed_data = transformed_data.select("asin","overall","reviewerID","reviewText","cosine_similarity")
    transformed_data.dropDuplicates()
    w2 = Window.partitionBy("reviewerID").orderBy(col("overall"))
    transformed_data = transformed_data.withColumn("row",row_number().over(w2)).filter(col("row") == 1).drop("row")

    top_10_rows = transformed_data.orderBy(F.desc("overall"), F.desc("cosine_similarity")).limit(10)
    top_10_rows = top_10_rows.withColumn("productURL",F.concat(lit('https://www.amazon.com/dp/'),F.col('asin')))
    top_10_rows = top_10_rows.select("asin","productURL","overall","reviewerID","reviewText","cosine_similarity")
    top_10_rows.show(10,truncate=True)
    return top_10_rows

#find top 10 review by TFIDF

def getTop10ReviewByTFIDF(numTopWords, review_data, questionText, outputPath):
    print('\nPreparing TF_IDF pipeline')
    pipeline_tfidf = createTFIDFPipeline(numTopWords)

    print('\nFit review data using TFIDF pipeline')
    data_model = pipeline_tfidf.fit(review_data)

    print('\nTransform review data using TFIDF pipeline')
    transformed_data = data_model.transform(review_data)

    df_top10 = getTop10Review(data_model, transformed_data, questionText)
    df_top10.write.format("json").save(outputPath)
    return df_top10


#find top 10 review by word2vec
def getTop10ReviewByWord2Vec(review_data,questionText, outputPath):
    print('\nPreparing WORD2VEC pipeline')
    # Split data into words
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    # Train Word2Vec model
    word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="words", outputCol="features")
    pipeline_word2vec = Pipeline(stages=[tokenizer,word2Vec])
    print('\nFit review data using WORD2VEC pipeline')
    data_model = pipeline_word2vec.fit(review_data)
    # Transform data with Word2Vec model
    print('\nTransform review data using WORD2VEC pipeline')
    word2vec_result = data_model.transform(review_data)
    # Output the results
    df_top10 = getTop10Review(data_model, word2vec_result, questionText)
    df_top10.write.format("json").save(outputPath)
    return df_top10

#find top10 review by BERT
def getTop10ReviewByBERT(review_data,questionText, outputPath):
    print('\nPreparing BERT pipeline')
    documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    sentence = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")

    embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128") \
        .setInputCols(["sentence"]) \
        .setOutputCol("sentence_bert_embeddings")\
        .setCaseSensitive(True) \
        .setMaxSentenceLength(512)

    embeddingsFinisher = EmbeddingsFinisher() \
        .setInputCols("sentence_bert_embeddings") \
        .setOutputCols("features") \
        .setOutputAsVector(True)

    # Create a Spark NLP pipeline
    pipeline = Pipeline(stages=[documentAssembler,
                            sentence,
                            embeddings,
                            embeddingsFinisher])

    print('\nFit review data using BERT pipeline')
    model = pipeline.fit(review_data)
    print('\nTransform review data using BERT pipeline')
    result_bert = model.transform(review_data)    
    # Output the results
    df_top10 = getTop10Review(model, result_bert, questionText, True)
    df_top10.write.format("json").save(outputPath)
    return df_top10

def getTrasformedData(dfReview):
    print("Convert userId and itemId columns from string to integer using StringIndexer")

    user_indexer = StringIndexer(inputCol="reviewerID", outputCol="reviewerIndex")
    item_indexer = StringIndexer(inputCol="asin", outputCol="asin_index")

    model_userIndexer = user_indexer.fit(dfReview) 
    indexed_df = model_userIndexer.transform(dfReview)
    model_itemIndexer = item_indexer.fit(indexed_df)
    indexed_df = model_itemIndexer.transform(indexed_df)
    return (model_userIndexer, model_itemIndexer ,indexed_df)

def trainALSModel(sc, dfReview, outputFolder):
    (model_userIndexer, model_itemIndexer ,indexed_df) = getTrasformedData(dfReview)
    print("Splitting train test data")
    (training_data, test_data) = indexed_df.randomSplit([0.8, 0.2], seed=1234)

    errors = []

    print("\nStart training model")
    for eachIter in range(10,31):
        # Create an ALS model
        als = ALS(maxIter=eachIter, regParam=0.005, userCol="reviewerIndex", itemCol="asin_index", 
                ratingCol="overall", coldStartStrategy="drop")

        # Fit the model to the training data
        model = als.fit(training_data)
        # Evaluate the model by computing the RMSE on the test data
        predictions = model.transform(test_data)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="overall",
                                    predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        errors.append(rmse)
        print(f"Iter {eachIter} , RMSE: {rmse:.4f}")
    
    print(f"\nAll errors: {errors}")

    x = np.array(range(10,31))
    y = np.array(errors)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Iteration', ylabel='RMSE',
        title='RMSE against each Iteration')
    ax.grid()
    ax.set_xlim(10, 30)
    ax.set_ylim(3.5, 5.5)
    print("\Saving Error plot")
    imagePath = f"{outputFolder.format('output_image')}/errorPlot.png"
    fig.savefig(imagePath)
    print("\Saving Final model")
    model.save(sc,outputFolder.format("output_ALS"))

def getrecomendation(dfReview, userId, save=False, ouputPath = ''):
    (model_userIndexer, model_itemIndexer ,indexed_df) = getTrasformedData(dfReview)
    als = ALS(maxIter=20, regParam=0.005, userCol="reviewerIndex", itemCol="asin_index", 
                ratingCol="overall", coldStartStrategy="drop")

        # Fit the model to the training data
    model = als.fit(indexed_df)
    # Generate recommendations for all users
    user_recs = model.recommendForAllUsers(10)  # Generate top 10 recommendations for each user

    # convert the recommendations to multiple rows per user with one recommendation in each row
    user_recs = user_recs.selectExpr("reviewerIndex", "explode(recommendations) as recommendations")

    # convert the recommendations column from {asin, rating} to columns productId  and rating
    user_recs = user_recs.selectExpr("reviewerIndex", "recommendations.asin_index as asin_index", 
                                    "recommendations.rating as rating")

    user_text = IndexToString(inputCol="reviewerIndex", outputCol="reviewerID", labels=model_userIndexer.labels)
    item_text = IndexToString(inputCol="asin_index", outputCol="asin", labels=model_itemIndexer.labels)

    user_recs_userText = user_text.transform(user_recs)
    user_recs_converted = item_text.transform(user_recs_userText)
    user_recs_converted = user_recs_converted.drop("reviewerIndex").drop("asin_index")

    if(save):
        user_recs_converted.write.format('csv').save(ouputPath.format("ALS_recommendation.csv"))

    user_recs_converted = user_recs_converted.filter(user_recs_converted.reviewerID == userId)
    return user_recs_converted