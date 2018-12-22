import docx
import unicodedata
import numpy
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
import pickle

doc = docx.Document('BCLsDataSet.docx')
fullText = []
for para in doc.paragraphs:
    fullText.append(para.text)
newf=[]
a=['Knowledge','','Comprehension','Application','Analysis','Synthesis','Evaluation']
for i in range(0,len(fullText)):
	fullText[i]=unicodedata.normalize('NFKD', fullText[i]).encode('ascii','ignore')
	if fullText[i] in a:
		continue
	else:
		newf.append(fullText[i])
vocab_size = 5000
encoded_docs = [one_hot(d, vocab_size) for d in newf]
max_review_length = 50
X_train = sequence.pad_sequences(encoded_docs, maxlen=max_review_length) 
with open('objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump(X_train, f)






