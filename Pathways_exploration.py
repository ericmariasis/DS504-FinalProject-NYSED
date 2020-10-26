#!/usr/bin/env python


#Tree methods Example
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier,GBTClassifier,RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from sklearn import metrics,preprocessing
from sklearn.decomposition import PCA

from collections import Counter
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer,load_boston,load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score,confusion_matrix

spark = SparkSession.builder.appName('rf').getOrCreate()


# In[47]:


# # Load training data
data = spark.read.csv('../../Cohort_Pathways_2019.csv',inferSchema=True,header=True)


# In[48]:


data.printSchema()


# In[49]:


data.head()


# In[50]:


data.describe().show()


# In[51]:


# In[52]:


data.columns

dataPandas = data.toPandas()

dataPandas.shape

data = data.na.drop()

## EDA
dataPandas = data.toPandas()

dataPandas.shape

dataPandas.info()

dataPandas["COURSE_OF_STUDY"].unique()

dataPandas["COURSE_OF_STUDY"].describe()

dataPandas["BOCES_NAME"].unique()

dataPandas["BOCES_NAME"].describe()

dataPandas["LEA_NAME"].unique()

dataPandas["LEA_NAME"].describe()

dataPandas["COUNTY_NAME"].unique()

dataPandas["COUNTY_NAME"].describe()

## light GBT
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# # Encode labels in column 'species'. 
# df['species']= label_encoder.fit_transform(df['species']) 
  
# df['species'].unique() 
cat_feats = ["REPORT_SCHOOL_YEAR","AGGREGATION_TYPE","AGGREGATION_NAME","LEA_NAME","NRC_DESC","BOCES_NAME",
          "COUNTY_NAME","MEMBERSHIP_DESC","SUBGROUP_NAME","COURSE_OF_STUDY_CODE","COURSE_OF_STUDY"]
for i in cat_feats:
    dataPandas[i] = dataPandas[i].astype('category')
    dataPandas[i] = label_encoder.fit_transform(dataPandas[i])

Y1 = dataPandas["COURSE_OF_STUDY"]
X = dataPandas.drop('COURSE_OF_STUDY',axis=1)
print(X.columns)
X_train,X_test,y_train,y_test=train_test_split(X,Y1,test_size=0.3,random_state=0)
clf = lgb.LGBMClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(y_pred)
print(metrics.classification_report(y_test,y_pred))
# view accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
## END lightgbt


assembler = VectorAssembler(
  inputCols=[
  #'REPORT_SCHOOL_YEAR', #string
  'AGGREGATION_INDEX', 
  #'AGGREGATION_TYPE', #string
  'AGGREGATION_CODE', 
  #'AGGREGATION_NAME', #string
  'LEA_BEDS', 
  #'LEA_NAME', #string
  'NRC_CODE', 
  #'NRC_DESC', #string
  'COUNTY_CODE', 
  #'COUNTY_NAME', #string
  'NYC_IND', 
  'BOCES_CODE', 
  #'BOCES_NAME', #string,dontworry
  'MEMBERSHIP_CODE', 
  'MEMBERSHIP_KEY', 
  #'MEMBERSHIP_DESC', #string
  'SUBGROUP_CODE', 
  #'SUBGROUP_NAME', #string
  #'COURSE_OF_STUDY_CODE', #string,dontworry
  #'COURSE_OF_STUDY', #string
  'STUDENT_COUNT'],
outputCol="features")

finoutlist = [
  'AGGREGATION_INDEX', 
  'AGGREGATION_TYPE_NUM', #string
  'AGGREGATION_CODE', 
  'AGGREGATION_NAME_NUM', #string
  'LEA_BEDS', 
  'LEA_NAME_NUM', #string
  'NRC_CODE', 
  'NRC_DESC_NUM', #string
  'COUNTY_CODE', 
  'COUNTY_NAME_NUM', #string
  'NYC_IND', 
  'BOCES_CODE', 
  'MEMBERSHIP_CODE', 
  'MEMBERSHIP_KEY', 
  'MEMBERSHIP_DESC_NUM', #string
  'SUBGROUP_CODE', 
  'SUBGROUP_NAME_NUM', #string
  'COURSE_OF_STUDY_NUM', #string
  'STUDENT_COUNT']

output = assembler.transform(data)

indexerAgtype = StringIndexer(inputCol="AGGREGATION_TYPE", outputCol="AGGREGATION_TYPE_NUM")
output = indexerAgtype.fit(output).transform(output)
output.select("AGGREGATION_TYPE","AGGREGATION_TYPE_NUM").show(truncate=False)

indexerAgname = StringIndexer(inputCol="AGGREGATION_NAME", outputCol="AGGREGATION_NAME_NUM")
output = indexerAgname.fit(output).transform(output)
output.select("AGGREGATION_NAME","AGGREGATION_NAME_NUM").show(truncate=False)

indexerleaname = StringIndexer(inputCol="LEA_NAME", outputCol="LEA_NAME_NUM")
output = indexerleaname.fit(output).transform(output)
output.select("LEA_NAME","LEA_NAME_NUM").show(truncate=False)

indexernrcdesc = StringIndexer(inputCol="NRC_DESC", outputCol="NRC_DESC_NUM")
output = indexernrcdesc.fit(output).transform(output)
output.select("NRC_DESC","NRC_DESC_NUM").show(truncate=False)

indexercountyname = StringIndexer(inputCol="COUNTY_NAME", outputCol="COUNTY_NAME_NUM")
output = indexercountyname.fit(output).transform(output)
output.select("COUNTY_NAME","COUNTY_NAME_NUM").show(truncate=False)

indexermembershipdesc = StringIndexer(inputCol="MEMBERSHIP_DESC", outputCol="MEMBERSHIP_DESC_NUM")
output = indexermembershipdesc.fit(output).transform(output)
output.select("MEMBERSHIP_DESC","MEMBERSHIP_DESC_NUM").show(truncate=False)

indexersubgroupname = StringIndexer(inputCol="SUBGROUP_NAME", outputCol="SUBGROUP_NAME_NUM")
output = indexersubgroupname.fit(output).transform(output)
output.select("SUBGROUP_NAME","SUBGROUP_NAME_NUM").show(truncate=False)

indexercoursestudyname = StringIndexer(inputCol="COURSE_OF_STUDY", outputCol="COURSE_OF_STUDY_NUM")
output = indexercoursestudyname.fit(output).transform(output)
output.select("COURSE_OF_STUDY","COURSE_OF_STUDY_NUM").show(truncate=False)

output = output.drop("features")

assembler = VectorAssembler(
  inputCols=[
  'AGGREGATION_INDEX', 
  'AGGREGATION_TYPE_NUM', #string
  'AGGREGATION_CODE', 
  'AGGREGATION_NAME_NUM', #string
  'LEA_BEDS', 
  'LEA_NAME_NUM', #string
  'NRC_CODE', 
  'NRC_DESC_NUM', #string
  'COUNTY_CODE', 
  'COUNTY_NAME_NUM', #string
  'NYC_IND', 
  'BOCES_CODE', 
  'MEMBERSHIP_CODE', 
  'MEMBERSHIP_KEY', 
  'MEMBERSHIP_DESC_NUM', #string
  'SUBGROUP_CODE', 
  'SUBGROUP_NAME_NUM', #string
  #'COURSE_OF_STUDY_NUM', #string
  'STUDENT_COUNT'],
  outputCol="features")

output = assembler.transform(output)

outputPandas = output.toPandas()

oCorr = outputPandas.corr()
sns.heatmap(oCorr)
plt.show()


## Showing feature correlations, heat map produced earlier
for i in range(len(finoutlist)):
   for j in range(len(finoutlist)):
       if i == j or i > j:
           continue
       thecorr = output.stat.corr(finoutlist[i],finoutlist[j])
       if abs(thecorr) >= 0.7:
           print("Corr between " + finoutlist[i] + " and " + finoutlist[j] + " is " + str(thecorr))
           output.select(finoutlist[i],finoutlist[j]).show(n=2,truncate=False)
           sns.lmplot(finoutlist[i],finoutlist[j],outputPandas)
           plt.show()


## ML


final_data = output.select("features",'COURSE_OF_STUDY_NUM')

train_data, test_data = final_data.randomSplit([0.7,0.3])
#train_data,test_data = final_data.random_split([0.8,0.2])

# Decision tree
# default: maxBins=32,maxDepth=5
# Random forest
# default: numTrees=20, maxDepth = 5,subsamplingRate=1.0,minInstancesPerNode=1,maxBins=32,featureSubsetStrategy='auto',minWeightFractionPerNode=0.0
#dtc = DecisionTreeClassifier(labelCol='COURSE_OF_STUDY_NUM',featuresCol='features')
#dtc = DecisionTreeClassifier(labelCol='COURSE_OF_STUDY_NUM',featuresCol='features',maxDepth=30,maxBins=70)

    # metricName = Param(Params._dummy(), "metricName",
    #                    "metric name in evaluation "
    #                    "(f1|accuracy|weightedPrecision|weightedRecall|weightedTruePositiveRate| "
    #                    "weightedFalsePositiveRate|weightedFMeasure|truePositiveRateByLabel| "
    #                    "falsePositiveRateByLabel|precisionByLabel|recallByLabel|fMeasureByLabel| "
    #                    "logLoss|hammingLoss)",
    #                    typeConverter=TypeConverters.toString)

from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
#train_datanb, test_datanb = final_data.randomSplit([0.7,0.3])
nb = NaiveBayes(smoothing=1,labelCol="COURSE_OF_STUDY_NUM")
modelnb = nb.fit(train_data)
predictionsnb = modelnb.transform(test_data)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="COURSE_OF_STUDY_NUM",metricName="weightedPrecision")
evaluator.evaluate(predictionsnb)


data_dict_acc_dtree = {}
data_dict_acc_dtree['maxDepth'] = []
data_dict_acc_dtree['accuracy'] = []
data_dict_acc_rf = {}
data_dict_acc_rf['num_trees'] = []
data_dict_acc_rf['accuracy'] = []
for i in range(2,9):
    dtc = DecisionTreeClassifier(labelCol='COURSE_OF_STUDY_NUM',featuresCol='features',maxDepth=i,maxBins=2000)
    dtc_model = dtc.fit(train_data)
    dtc_predictions = dtc_model.transform(test_data)
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="COURSE_OF_STUDY_NUM", predictionCol="prediction", metricName="accuracy")
    dtc_acc = acc_evaluator.evaluate(dtc_predictions)
    data_dict_acc_dtree['maxDepth'].append(i)
    data_dict_acc_dtree['accuracy'].append(dtc_acc)
    print("Here are the results!")
    print('-'*80)
    print('A single decision tree had an accuracy of: {0:2.2f}%'.format(dtc_acc*100))


for i in [13,15,17]:
#for i in range(1,5):
    rfc = RandomForestClassifier(labelCol='COURSE_OF_STUDY_NUM',featuresCol='features',maxDepth=7, numTrees=i,maxBins=2000)
    rfc_model = rfc.fit(train_data)
    rfc_predictions = rfc_model.transform(test_data)
    y_test = rfc_predictions.select("COURSE_OF_STUDY_NUM").toPandas()
    y_pred = rfc_predictions.select("prediction").toPandas()
    confusion_matrix(y_test,y_pred)
    print(metrics.confusion_matrix(y_test,y_pred))
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="COURSE_OF_STUDY_NUM", predictionCol="prediction", metricName="f1")
    rfc_acc = acc_evaluator.evaluate(rfc_predictions)
    data_dict_acc_rf['num_trees'].append(i)
    data_dict_acc_rf['accuracy'].append(rfc_acc)
    print('-'*80)
    print('A random forest had an accuracy of: {0:2.2f}%'.format(rfc_acc*100))

data_dict_acc_dtree_pd = pd.DataFrame.from_dict(data_dict_acc_dtree)
sns.relplot(x='maxDepth',y='accuracy',data=data_dict_acc_dtree_pd,markers=True)
plt.title("Decision Tree Accuracy vs Max Depth")
plt.scatter(data_dict_acc_dtree['maxDepth'],data_dict_acc_dtree['accuracy'])
plt.plot(data_dict_acc_dtree['maxDepth'],data_dict_acc_dtree['accuracy'])
plt.show()
data_dict_acc_rf_pd = pd.DataFrame.from_dict(data_dict_acc_rf)
sns.relplot(x='num_trees',y='accuracy',data=data_dict_acc_rf_pd,markers=True)
plt.title("Random Forest Accuracy vs Number of Trees")
plt.scatter(data_dict_acc_rf['num_trees'],data_dict_acc_rf['accuracy'])
plt.plot(data_dict_acc_rf['num_trees'],data_dict_acc_rf['accuracy'])
plt.show()

#rfc_model = rfc.fit(final_data)
rfc_model.featureImportances
# # gradient boosting
# gbt = GBTClassifier(labelCol='COURSE_OF_STUDY_NUM',featuresCol='features')
# gbt_model = gbt.fit(train_data)
# gbt_predictions = gbt_model.transform(test_data)
# acc_evaluator = MulticlassClassificationEvaluator(labelCol="COURSE_OF_STUDY_NUM", predictionCol="prediction", metricName="accuracy")
# gbt_acc = acc_evaluator.evaluate(gbt_predictions)
# print('-'*80)
# print('A ensemble using GBT had an accuracy of: {0:2.2f}%'.format(gbt_acc*100))
