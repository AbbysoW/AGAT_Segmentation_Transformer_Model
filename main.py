from Transformer import Transformer

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import fasttext
import fasttext.util

# Constants
NUM_ENCODERS = 2
NUM_DECODERS = 0
D_MODEL = 296
NUM_HEADS = 2
D_FF = 128
OUTPUT_VACABULARY_SIZE = 5
SEQ_LEN = 50

BATCH_SIZE = 32
EPOCHS = 50

FT = fasttext.load_model('wiki.ru.bin')  # Uses russian fasttext vectors 

# DataSet Prepearing
data = pd.DataFrame()
data = pd.read_csv("Segmentation_DataSet_v0.2.csv", delimiter=';')

nltk.download('punkt_tab')
nltk.download('stopwords')

punctuation_marks = ['.', '..', '...', '!', '?', ',', '(', ')', '<', '>', '«', '»', '{', '}', '[', ']',':', ';', '_', '-', '–', '%', '@', '#', '№', '^', '&', '+', '=', '*', '/', '|', '"', "'", '~']

def preprocess(text, punctuation_marks): # Splits text into  tokens
  preprocessed_text = []
  if type(text) == str:
    tokens = word_tokenize(text.lower())
    for token in tokens:
      if token not in punctuation_marks:
          preprocessed_text.append(token)
  return preprocessed_text[:SEQ_LEN]

class_counts = [0, 0, 0, 0, 0]
def output_proccess(mask): # Splits uotput into tokens and converts into one_hot encoding
  global class_counts
  mask = list(mask)
  mask = [to_categorical(int(item), 5) for item in mask if item != ' ']
  for i in range(SEQ_LEN - len(mask)):
    mask.append(to_categorical(0, 5))
  for i in mask:
    class_counts[np.argmax(i)] += 1
  return mask[:SEQ_LEN]

def word_to_vec(text, embading): # Convets token into vector
  preprocessed_text = []
  for token in text:
    preprocessed_text.append(embading.get_word_vector(token)[:D_MODEL])
  for i in range(SEQ_LEN - len(preprocessed_text)):
     preprocessed_text.append(np.zeros(D_MODEL))
  return preprocessed_text

def padding_proocess(output): # Prepears padding mask 
  output = [item for item in output if item != ' ']
  padding_mask = [1 for item in output]
  for i in range(SEQ_LEN - len(output)):
    padding_mask.append(0)
  padding_mask = np.array(padding_mask)    
  padding_mask  = padding_mask[None, None, :]
  return padding_mask

data['DataTokens'] = data.apply(lambda row: preprocess(row['Data'], punctuation_marks), axis=1)
data['DataVectors'] = data.apply(lambda row: word_to_vec(row['DataTokens'], FT), axis=1)
data['OutputTokens']  = data.apply(lambda row: output_proccess(row['Output']), axis=1)
data['PaddingMask'] =  data.apply(lambda row: padding_proocess(row['Output']), axis=1)

total = sum(class_counts)
class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
class_weights

x_train, x_test, y_train, y_test, padding_train, padding_test = train_test_split(data['DataVectors'], data['OutputTokens'], data['PaddingMask'], train_size=0.75, random_state = 42)

x_train = np.array(x_train.tolist(), dtype=np.float32)
x_test = np.array(x_test.tolist(), dtype=np.float32)
padding_train = np.array(padding_train.tolist(), dtype=np.float32)

y_train = np.array(y_train.tolist(), dtype=np.int16)
y_test = np.array(y_test.tolist(), dtype=np.int16)
padding_test = np.array(padding_test.tolist(), dtype=np.float32)

input = {
    "encoder_input": x_train,
    "decoder_input": np.zeros((len(x_train), SEQ_LEN, D_MODEL)),
    # "attention_mask": None,
    "encoder_padding_mask": padding_train,
    # "decoder_padding_mask": None
}

dataset = tf.data.Dataset.from_tensor_slices((
    input,
    y_train
))
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

validation_input = {
    "encoder_input": x_test,
    "decoder_input": np.zeros((len(x_test), SEQ_LEN, D_MODEL)),
    # "attention_mask": None,
    "encoder_padding_mask": padding_test,
    # "decoder_padding_mask": None
}

validation_dataset = tf.data.Dataset.from_tensor_slices((
    validation_input,
    y_test
))
validation_dataset = validation_dataset.batch(BATCH_SIZE)

# Model
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9
)
transformer  = Transformer(
    NUM_ENCODERS,
    NUM_DECODERS,
    D_MODEL,
    NUM_HEADS,
    D_FF,
    OUTPUT_VACABULARY_SIZE,
    SEQ_LEN,
    dropout_rate = 0.1
    )
transformer.compile(
    optimizer= optimizer,
    loss= keras.losses.CategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy'],
    weighted_metrics=['accuracy']
    )

model_cnn_save_path = 'MainModel.h5'
checkpoint_callback_cnn = ModelCheckpoint(model_cnn_save_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)

history = transformer.fit(
    dataset,
    epochs= EPOCHS,
    validation_data= validation_dataset,
    class_weight= class_weights
    )