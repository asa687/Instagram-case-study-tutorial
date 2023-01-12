"""
dataset from statso

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor



data = pd.read_csv("Instagram data.csv", encoding = 'latin1') 
## this drops null values from the dataset 
data = data.dropna()  
print(data.info())  

## this draws a graph showing impressions from Home
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Comments")
sns.distplot(data['Comments'])
plt.show()  

## this draws a pie chart representing impressions
fromHome = data["From Home"].sum()
fromHashtags = data["From Hashtags"].sum()
fromExplore = data["From Explore"].sum()
fromOther = data["From Other"].sum() 
labels = ['From Home','From Hashtags','From Explore','Other']
values = [fromHome, fromHashtags, fromExplore, fromOther]

impressionPiechart = px.pie(data, values=values, names=labels, title='Impressions on Instagram Posts From Various Sources', hole=0.5)
impressionPiechart.show() 

## this produces a corelation table for likes and the other categories
correlation = data.corr()
print(correlation["Likes"].sort_values(ascending=False))

## this produces a conversion rate (percentage of visitors who become followers)
conversionRate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100 
print(conversionRate) 

##this splits the data into training and test sets. the aim of the model that is to be created is to predict impressions
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42) 

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest) 


