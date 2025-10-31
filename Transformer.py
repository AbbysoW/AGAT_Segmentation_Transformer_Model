import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, LayerNormalization
from tensorflow.keras.models import Model
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
    
    def attention_function(self, q, k, v):
        """Высчитывет влияние каждоо слова в заданом предложении\n
        q - матрица запросов, k - матрица ключей, v - матрица значений"""
        matrix_mult = tf.matmul(q, k, transpose_b= True)
        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_logits = matrix_mult / tf.math.sqrt(d_k)

        attention_weights = tf.nn.softmax(scaled_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)

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

class TransformerBlock(tf.keras.layers.Layer):
    """Блок трансформера в  котором input швуе в слой MultiHeadAttention после нормализуется и идет в Feedforward после которог слнова нормализуется\n
    d_model - размерность эмбэдинга модели, num_heads - количество голов, d_ff - рамерность входного слоя в Feedforward, dropout_rate - значение откланения"""
    def __init__(self, d_model:int, num_heads:int, d_ff:int, dropout_rate:float=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = Feedforward(d_model, d_ff)

        self.layernorm1 = LayerNormalization(epsilon= 1e-6)
        self.layernorm2 = LayerNormalization(epsilon= 1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training:bool=False, mask= None):
        attention_output = self.att(x, x, x)
        attention_output = self.dropout1(attention_output, training=training)
        attention_result = self.layernorm1(x + attention_output)

        feedforward_output = self.feedforward(attention_result)
        feedforward_output = self.dropout2(feedforward_output, training=training)
        
        output = self.layernorm2(attention_result + feedforward_output)
        return output


class Encoder(tf.keras.layers.Layer):
    """Блок энкодер принемающий на вход запрос и отпраляющий его на TransformerBlock\n
    d_model - размерность эмбэдинга модели, num_layers - количество энкодинг слоев, num_heads - количество голов, d_ff - рамерность входного слоя в Feedforward, input_vocab_size - кол-во векторов во входном эмбэдинге, maximum_position_encoding - длина контекстного словаря, dropout_rate - значение откланения"""
    def __init__(self, d_model:int, num_layers:int, num_heads:int, d_ff:int, input_vocab_size:int, maximum_position_encoding:int, dropout_rate:float=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dropout = Dropout(dropout_rate)

        self.encoding_layers = [TransformerBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

    def call(self, x, training:bool=False, mask=None):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoding_layers[i](x, training=training, mask=mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """Блок декодер принемающий на вход значения из енкодера и отпраляющий его на TransformerBlock\n
    d_model - размерность эмбэдинга модели, num_layers - количество энкодинг слоев, num_heads - количество голов, d_ff - рамерность входного слоя в Feedforward, output_vocab_size - кол-во векторов в выходном эмбэдинге, maximum_position_encoding - длина контекстного словаря, dropout_rate - значение откланения"""
    def __init__(self, d_model:int, num_layers:int, num_heads:int, d_ff:int, output_vocab_size:int, maximum_position_encoding:int, dropout_rate:float=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = Embedding(output_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dropout = Dropout(dropout_rate)

        self.decoding_layers = [TransformerBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

    def call(self, x, training:bool=False, look_ahead_mask= None, padding_mask= None):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.decoding_layers[i](x, training=training, mask= look_ahead_mask)

        return x


class Transformer(Model):

    name = None

    def __init__(self, num_layers:int, d_model:int, num_heads:int, d_ff:int, input_vocab_size:int, output_vocab_size:int, maximum_position_encoding:int, dropout_rate:float= 0.1):
        super().__init__()

        self.encoder = Encoder(d_model, num_layers, num_heads, d_ff, input_vocab_size, maximum_position_encoding, dropout_rate)

        self.decoder = Decoder(d_model, num_layers, num_heads, d_ff, output_vocab_size, maximum_position_encoding, dropout_rate)

        self.final_layer = Dense(output_vocab_size)

    def call(self, x, y, training:bool=False, look_ahead_mask=None, padding_mask=None):
        
        enc_output = self.encoder(x, training=training, mask=padding_mask)

        dec_output, _ = self.decoder(y, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output
    
    def __str__(self):
        return f"Name: {self.name}\nEncoding: \n\tnumber: {self.encoder.num_layers}\nDecoding: \n\tnumber:{self.decoder.num_layers}\nInput_size: {input_vocab_size}\nOutput_size: {output_vocab_size}"
    

if __name__ == "__main__":

    num_layers = 2
    d_model = 296
    num_heads = 8
    d_ff = 128
    input_vocab_size = 100
    output_vocab_size = 100
    maximum_position_encoding = 40

    transformer  = Transformer(
        num_layers, 
        d_model, 
        num_heads, 
        d_ff, 
        input_vocab_size, 
        output_vocab_size, 
        maximum_position_encoding
        )
    
    transformer.name = "Test_model"
    
    print(transformer)
