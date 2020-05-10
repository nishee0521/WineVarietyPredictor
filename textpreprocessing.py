import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler


def TextPreProcessing(DataFrame):
	#concatenate all the categorial variables in a single column to make a dictionary
	DataFrame['AllFeatures'] = DataFrame.apply(lambda x: x['review_title'] + x['review_description'] + x['country'] + x['province'] + x['region_1'] + x['winery'] + x['designation'] if pd.notnull(x['designation']) else x['review_title'] + x['review_description'] + x['country'] + x['province'] + x['region_1'] + x['winery'], axis=1)

	#with the help of tokenizer created a dictionary of all the words in the given dataset
	token= Tokenizer(num_words=100000,oov_token="<oov>")
	token.fit_on_texts(df['AllFeatures']) 

	mytext=token.texts_to_sequences(df['AllFeatures'])
	ptext=pad_sequences(mytext,maxlen=100,padding="post",truncating="post")


	#One hot encoding the dependent variable: variety
	train=pd.get_dummies(train, columns=['variety'])

	#converting pice column into a numpy array
	price=DataFrame['price'].to_numpy()
	price=price.reshape(-1,1)

	#converting points coulmn to numpy array
	points=DataFrame['points'].to_numpy()
	points=points.reshape(-1,1)

	#seperating the dependent variable: variety
	y=train.loc[:, "variety_Bordeaux-style Red Blend":].values
	y=y.astype("float")

	#NORMALIZING THE FEATURES
	#PRICE:
	sc=MinMaxScaler()
	price=sc.fit_transform(price)

	#POINTS:
	sclr=MinMaxScaler()
	points=sclr.fit_transform(points)

	#PTEXT:
	scaler=MinMaxScaler()
	ptext=scaler.fit_transform(ptext)

	#Concatenate price, points and ptext as a single input for model
	final=np.concatenate((ptext,price,points),axis=1)

	fin_train=final[:len(train_df),:]
	fin_test=final[len(train_df):,]

	return fin_train, fin_test



