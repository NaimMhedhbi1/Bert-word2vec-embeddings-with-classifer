from Utilis.cleaning import clean_text , Preprocess_list_french
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from Utilis.config import MAX_SEQUENCE_LENGTH
from Utilis.load_model import model 

def predict(model, review, seq_length = MAX_SEQUENCE_LENGTH):
    
    tokenizer = Tokenizer()
    review = clean_text(review)
    review = Preprocess_list_french(review, dicto='data/dictionnaire.txt')
    tokenizer.fit_on_texts(review)
    word_index = tokenizer.word_index
    review_padded = pad_sequences(tokenizer.texts_to_sequences(review),
                        maxlen = seq_length)

    
    if(len(review_padded) == 0):
        "Your review must contain at least 1 word!"
        return None
    
    score = model.predict(review_padded)
    y_pred = [np.argmax(score)]
    print('the category is : ' + str(y_pred) )
    
    return y_pred 


