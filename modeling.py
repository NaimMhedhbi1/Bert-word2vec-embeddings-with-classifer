from Utilis import *
from loguru import logger


__author__ = "NaimMhedhbi1"


PATH_train = 'data/train_datam.csv'
PATH_test = 'data/test_datam.csv'

logger.info("reading the dataset")
train_data = pd.read_csv(PATH_train)  
train_data['text_clean'] = train_data['pname'].apply(clean_text) 
preprocess_list  = Preprocess_list_french(train_data['texte_clean'],dicto='data/dictionnaire.txt')
train_data['text_clean'] = preprocess_list 