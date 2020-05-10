from preprocessing import*
from textpreprocessing import *
from models import *

def main():

	complete_df,test_df,varieties=DataPreProcessing(train.csv,test.csv)

	fin_train,fin_test=TextPreProcessing(complete_df)

	classifier=model(fin_train,fin_test)

	y_pred = classifier2.predict(fin_test.reshape(-1,102,1))
	yactual=np.zeros(y_pred.shape)
	another = np.argmax(y_pred, axis=1)
	yactual[np.arange(another.size),another]=1

	
	vv=[]
	for i in range(len(varieties)):
	  vv.append(str(varieties[i]).split('_')[1])

	predictions = []
	for anoth in another:
	  predictions.append(vv[anoth])

	
	test_df['variety'] = predictions
	test_df.to_csv('test.csv')


