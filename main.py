from Utilis.cleaning import *
from Utilis.config import *
from Utilis.predict import *
from Utilis.load_model import *
from Utilis.config import MAX_SEQUENCE_LENGTH
from loguru import logger


__author__ = "NaimMhedhbi1"




logger.info("reading the training dataset")
train_data = pd.read_csv(PATH_train)
train_data['text'] = train_data['pname'].apply(clean_text)
preprocess_list = Preprocess_list_french(
    train_data['texte'], dicto='data/dictionnaire.txt')
train_data['text'] = preprocess_list


logger.info("reading the test dataset")
test_data = pd.read_csv(PATH_test)
test_data['text'] = test_data['pname'].apply(clean_text)
preprocess_list = Preprocess_list_french(
    test_data['texte'], dicto='data/dictionnaire.txt')
test_data['text'] = preprocess_list

#the Model already trained in the notebook, check : Model.ipynb. the model is saved. we only load it for the single prediction task. 
 

logger.info("single prediction")
review = 'My Eyes - Taille-crayons'
y_pred = predict(model, review, seq_length = MAX_SEQUENCE_LENGTH)