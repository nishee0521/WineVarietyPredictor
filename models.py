from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Reshape

def model(fin_train,fin_test):

	classifier2 = Sequential()
	classifier2.add(Conv1D(input_shape=(102,1),filters=32, kernel_size=3,strides=1, activation='relu', kernel_initializer='uniform'))
	classifier2.add(MaxPooling1D(pool_size=4,padding="same",strides=1))
	classifier2.add(Conv1D(filters=32, kernel_size=5,strides=1, activation='relu', kernel_initializer='uniform'))
	classifier2.add(MaxPooling1D(pool_size=4,padding="same",strides=1))
	classifier2.add(Flatten())
	# classifier.add(Dense(units=1024,activation='relu'))
	classifier2.add(Dense(units=512,activation='relu', kernel_initializer='uniform'))
	classifier2.add(Dense(units=512,activation='relu', kernel_initializer='uniform'))
	classifier2.add(Dense(units=256, activation='linear', kernel_initializer='uniform'))
	classifier2.add(Dense(units=28,activation='softmax', kernel_initializer='uniform'))




	classifier2.fit(fin_train.reshape(-1, 102,1), y.reshape(-1, 28), epochs=25, batch_size=256)

	return classifier2