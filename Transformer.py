import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Embedding, Dropout, LayerNormalization
from keras.models import Model
import numpy as np

def positional_encoding(d_model:int, max_position:int):
    """Генерирует графики Sin и cos в количестве d_model и выделяет из них точки через равные промежутки в размере max_position \n
    d_model - размерность эмбэдинга модели, max_position - максиальная позиция слова"""
    angle_rads = np.arange(max_position)[:, np.newaxis] / np.power(
    10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class MultiHeadAttention(tf.keras.layers.Layer):
    def  __init__(self, d_model:int, num_heads:int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)
    
    def split_heads(self, x, batch_size:int):
        """Разделяет исходную матрицу на головы размером depth в количестве num_heads\n
        num_heads - количество голов, depth - размерность каждой головы"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def attention_function(self, q, k, v, mask= None):
        """Высчитывет влияние каждоо слова в заданом предложении\n
        q - матрица запросов, k - матрица ключей, v - матрица значений"""
        matrix_mult = tf.matmul(q, k, transpose_b= True)
        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_logits = matrix_mult / tf.math.sqrt(d_k)

        if mask is not None:
            scaled_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, q, k, v, mask= None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.attention_function(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output
    
class Feedforward(tf.keras.layers.Layer):
    """Нейронная сеть из полносвязный слоев\n
    d_model - размерность выходного слоя, d_ff - размерность входного слоя"""
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.dense1 = Dense(d_ff, activation='relu')
        self.dense2 = Dense(d_model)

    def call(self, x):
        x = self.dense1(x)
        output = self.dense2(x)
        return output

class Encoder(tf.keras.layers.Layer):
    """Блок энкодер принемающий на вход запрос и отпраляющий его на TransformerBlock\n
    d_model - размерность эмбэдинга модели, num_layers - количество энкодинг слоев, num_heads - количество голов, d_ff - рамерность входного слоя в Feedforward, input_vocab_size - кол-во векторов во входном эмбэдинге, maximum_position_encoding - длина контекстного словаря, dropout_rate - значение откланения"""
    def __init__(self, d_model:int, num_heads:int, d_ff:int, input_vocab_size:int, maximum_position_encoding:int, dropout_rate:float=0.1):
        super().__init__()

        self.d_model = d_model

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = Feedforward(d_model, d_ff)

        self.layernorm1 = LayerNormalization(epsilon= 1e-6)
        self.layernorm2 = LayerNormalization(epsilon= 1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)


    def call(self, x, training:bool=False, mask=None):

        attention_output = self.attention(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        attention_result = self.layernorm1(x + attention_output)

        feedforward_output = self.feedforward(attention_result)
        feedforward_output = self.dropout2(feedforward_output, training=training)
        
        output = self.layernorm2(attention_result + feedforward_output)

        return output


class Decoder(tf.keras.layers.Layer):
    """Блок декодер принемающий на вход значения из енкодера и отпраляющий его на TransformerBlock\n
    d_model - размерность эмбэдинга модели, num_layers - количество энкодинг слоев, num_heads - количество голов, d_ff - рамерность входного слоя в Feedforward, output_vocab_size - кол-во векторов в выходном эмбэдинге, maximum_position_encoding - длина контекстного словаря, dropout_rate - значение откланения"""
    def __init__(self, d_model:int, num_heads:int, d_ff:int, output_vocab_size:int, maximum_position_encoding:int, dropout_rate:float=0.1):
        super().__init__()

        self.d_model = d_model

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = Feedforward(d_model, d_ff)

        self.layernorm1 = LayerNormalization(epsilon= 1e-6)
        self.layernorm2 = LayerNormalization(epsilon= 1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, enc_output= None, training:bool=False, look_ahead_mask= None, padding_mask= None):

        attention_output = self.enc_dec_attention(x, x, x, look_ahead_mask)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = self.layernorm1(x + attention_output)
        
        if enc_output == None:
            enc_output = attention_output
        
        enc_dec_attention_output = self.enc_dec_attention(attention_output, enc_output, enc_output)
        enc_dec_attention_output = self.dropout1(enc_dec_attention_output, training=training)
        enc_dec_attention_output = self.layernorm1(attention_output + enc_dec_attention_output)

        feedforward_output = self.feedforward(enc_dec_attention_output)
        feedforward_output = self.dropout2(feedforward_output, training=training)
        
        output = self.layernorm2(enc_dec_attention_output + feedforward_output)

        return output


class Transformer(Model):

    name = None

    def __init__(self, num_encoder_layers:int, num_decoder_layers:int, d_model:int, num_heads:int, d_ff:int, input_vocab_size:int, output_vocab_size:int, maximum_position_encoding:int, dropout_rate:float= 0.1):
        super().__init__()

        self.d_model = d_model

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        self.embedding = Embedding(output_vocab_size, d_model)
        self.pos_encoding = positional_encoding(d_model, maximum_position_encoding) 
        self.dropout = Dropout(dropout_rate)

        self.encoder = [Encoder(d_model, num_heads, d_ff, input_vocab_size, maximum_position_encoding, dropout_rate) for _ in range(num_encoder_layers)]

        self.decoder = [Decoder(d_model, num_heads, d_ff, output_vocab_size, maximum_position_encoding, dropout_rate) for _ in range(num_decoder_layers)]

        self.linear = Dense(output_vocab_size)
    
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, args= (None, None), training:bool=False, look_ahead_mask=None, padding_mask=None):
        x_encoder, x_decoder = args

        if self.num_encoder_layers == 0:
            x_encoder = None

        if self.num_decoder_layers == 0:
            x_decoder = None

        if x_encoder is not None:
            seq_len = tf.shape(x_encoder)[1]

            x_encoder += self.pos_encoding[:, :seq_len, :]
   
            x_encoder = self.dropout(x_encoder, training=training)

            for layer in self.encoder:
                x_encoder = layer(x_encoder, training=training, mask=padding_mask)
        
        if x_decoder is not None:
            seq_len = tf.shape(x_decoder)[1]

            x_decoder += self.pos_encoding[:, :seq_len, :]

            x_decoder = self.dropout(x_decoder, training=training)

            for layer in self.decoder:
                x_decoder = layer(x_decoder, x_encoder, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        else:
            x_decoder = x_encoder

        output = self.linear(x_decoder)

        print("done")
        return output


    def __str__(self):
        return f"Name: {self.name}\nEncoding: \n\tnumber: {self.num_encoder_layers}\nDecoding: \n\tnumber:{self.num_decoder_layers}\nInput_size: {input_vocab_size}\nOutput_size: {output_vocab_size}"
    

if __name__ == "__main__":

    num_encoders = 2
    num_decoders = 0
    d_model = 296
    num_heads = 8
    d_ff = 128
    input_vocab_size = 100
    output_vocab_size = 3
    maximum_position_encoding = 40

    transformer  = Transformer(
        num_encoders,
        num_decoders, 
        d_model, 
        num_heads, 
        d_ff, 
        input_vocab_size, 
        output_vocab_size, 
        maximum_position_encoding
        )
    
    transformer.name = "Test_model"
    
    print(transformer)

    # y = transformer(
    #     args = (np.ones((1, 40, 296)), np.zeros((1, 40, 296)))
    # )

    y = transformer.predict((np.ones((1, 40, 296)), np.zeros((1, 40, 296))))

    print(y)

