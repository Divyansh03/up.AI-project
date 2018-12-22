import numpy as np
import keras
import pickle
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout
from keras.layers import Bidirectional
from attention_decoder import AttentionDecoder


with open('objs.pkl') as f: 
    X_train = pickle.load(f)
y_train=[]
for i in range(0,4):
	for j in range(1,102):
		y_train.append(i)
for i in range(4,6):
	for j in range(1,101):
		y_train.append(i)
X_train, y_train = shuffle(X_train, y_train)
y_train = to_categorical(y_train, num_classes=None)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



embedding_vector_length = 16
model = Sequential() 
model.add(Embedding(5000, embedding_vector_length, input_length=50)) 
model.add(Bidirectional(LSTM(100)))
model.add(Activation('softmax')) #this guy here
model.add(Dropout(0.5))
#model.add(Dense(10, activation='relu')) 
model.add(Dense(6,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=100, batch_size=64) 

scores = model.evaluate(X_test, y_test, verbose=0) 
print("Accuracy: %.2f%%" % (scores[1]*100))



