import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, LayerNormalization, Embedding
from keras.models import Model
import numpy as np

def positional_encoding(d_model:int, seq_len:int):
    """
    Генерирует графики Sin и Cos в количестве d_model и выделяет из них точки через равные промежутки в количестве seq_len

    ---------------------------------------

    d_model - размерность эмбэдинга модели,
    seq_len - длина последовательности модели
    """
    angle_rads = np.arange(seq_len)[:, np.newaxis] / np.power(
    10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class MultiHeadAttention(keras.layers.Layer):
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
        """
        Разделяет исходную матрицу на головы размером self.depth в количестве self.num_heads

        ---------------------------------------

        num_heads - количество голов, 
        depth - размерность каждой головы
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def attention_function(self, q, k, v, mask= None):
        """
        Высчитывет влияние каждоо слова в заданом предложении

        ---------------------------------------

        q - матрица запросов, 
        k - матрица ключей, 
        v - матрица значений
        """
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
    
class Feedforward(keras.layers.Layer):
    """
    Нейронная сеть из полносвязный слоев

    ---------------------------------------

    d_model - размерность выходного слоя, 
    d_ff - размерность вsходного слоя
    """
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.dense1 = Dense(d_ff, activation='relu')
        self.dense2 = Dense(d_model)

    def call(self, x):
        x = self.dense1(x)
        output = self.dense2(x)
        return output

class Encoder(keras.layers.Layer):
    """
    Блок энкодер принемающий на вход запрос и отпраляющий его на TransformerBlock
    
    ---------------------------------------

    d_model - размерность эмбэдинга модели, 
    num_heads - количество голов, 
    d_ff - рамерность входного слоя в Feedforward, 
    dropout_rate - значение откланения
    """
    def __init__(self, d_model:int, num_heads:int, d_ff:int, dropout_rate:float=0.1):
        super().__init__()

        self.d_model = d_model

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = Feedforward(d_model, d_ff)

        self.layernorm1 = LayerNormalization(epsilon= 1e-6)
        self.layernorm2 = LayerNormalization(epsilon= 1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)


    def call(self, x, training:bool= False, padding_mask= None):

        if not padding_mask is None:
            inverted_padding_mask = 1 - padding_mask
        else:
            inverted_padding_mask = None

        attention_output = self.attention(x, x, x, inverted_padding_mask)

        # for matrix in attention_output:
        #     print(matrix[3][3],"|", end="")
        # print("")
        
       # tf.print("\nAfter Attention \n", attention_output[0][:3][:6])
       # tf.print(attention_output.shape)

        attention_output = self.dropout1(attention_output, training= training)
        attention_output = self.layernorm1(x + attention_output)

       # tf.print("After Attention Norm\n", attention_output[0][:3][:6])
       # tf.print(attention_output.shape)

        feedforward_output = self.feedforward(attention_output)

       # tf.print("\nAfter FeedForward\n", feedforward_output[0][:3][:6])
       # tf.print(feedforward_output.shape)

        feedforward_output = self.dropout2(feedforward_output, training= training)
        
        output = self.layernorm2(attention_output + feedforward_output)

        ## tf.print("After FeedForward Norm\n", output[0][:3][:6])
        ## tf.print(output.shape)

        return output


class Decoder(keras.layers.Layer):
    """
    Блок декодер принемающий на вход значения из енкодера и отпраляющий его на TransformerBlock
    
    ---------------------------------------

    d_model - размерность эмбэдинга модели, 
    num_heads - количество голов, 
    d_ff - рамерность входного слоя в Feedforward, 
    dropout_rate - значение откланения
    """
    def __init__(self, d_model:int, num_heads:int, d_ff:int, dropout_rate:float= 0.1):
        super().__init__()

        self.d_model = d_model

        self.attention = MultiHeadAttention(d_model, num_heads,)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = Feedforward(d_model, d_ff)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

        self.layernorm1 = LayerNormalization(epsilon= 1e-6)
        self.layernorm2 = LayerNormalization(epsilon= 1e-6)
        self.layernorm3 = LayerNormalization(epsilon= 1e-6)

    def call(self, x, enc_output= None, training:bool=False, attention_mask= None, padding_mask= None):

       # tf.print("\n\nDECODER DETECTED YOPTA\n\n")

        if not(attention_mask is None or padding_mask is None):
            universal_mask  = attention_mask * padding_mask
            inverted_universal_mask = 1 - universal_mask
        else:
            inverted_universal_mask = None

       # tf.print("\nAfter Attention \n", attention_output[0][:3][:6])

        attention_output = self.attention(x, x, x, inverted_universal_mask)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = self.layernorm1(x + attention_output)

       # tf.print("After Attention Norm\n", attention_output[0][:3][:6])
        
        if enc_output is None:
            enc_output = attention_output
        
        enc_dec_attention_output = self.enc_dec_attention(attention_output, enc_output, enc_output)
        enc_dec_attention_output = self.dropout2(enc_dec_attention_output, training=training)
        enc_dec_attention_output = self.layernorm2(attention_output + enc_dec_attention_output)

        feedforward_output = self.feedforward(enc_dec_attention_output)
        feedforward_output = self.dropout3(feedforward_output, training=training)
        
        output = self.layernorm3(enc_dec_attention_output + feedforward_output)

        return output


class Transformer(Model):

    name = None

    def __init__(self, num_encoder_layers:int, num_decoder_layers:int, d_model:int, num_heads:int, d_ff:int, 
                 output_vocab_size:int, seq_len:int, dropout_rate:float= 0.1, input_vocab_size:int= None, target_vocab_size:int= None):
        super().__init__()

        self.d_model = d_model

        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        self.pos_encoding = positional_encoding(d_model, seq_len) 
        self.dropout = Dropout(dropout_rate)
        
        if not input_vocab_size is None:
            self.encoder_embedding = Embedding(input_vocab_size, d_model)

        if not target_vocab_size is None:
            self.decoder_embedding = Embedding(target_vocab_size, d_model)


        self.encoder = [Encoder(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_encoder_layers)]

        self.decoder = [Decoder(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_decoder_layers)]

        self.linear = Dense(output_vocab_size, use_bias= True)
    
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, args, training:bool=False):

        if isinstance(args, dict):
            x_encoder = args.get('encoder_input')
            x_decoder = args.get('decoder_input')
            attention_mask = args.get('attention_mask')
            encoder_padding_mask = args.get('encoder_padding_mask')
            decoder_padding_mask = args.get('decoder_padding_mask')
        else:
            x_encoder, x_decoder = args
            attention_mask = None
            encoder_padding_mask = None
            decoder_padding_mask = None

        # print(f"PositionEncoding: {self.pos_encoding.shape}")

        # print(x_encoder.shape)
        # print(x_decoder.shape)

        # print(f"Number of encoder layers: {self.num_encoder_layers}")
        # print(f"Number of decoder layers: {self.num_decoder_layers}")
        # print(f"Model has {len(self.encoder)} encoder layers")
        # print(f"Model has {len(self.decoder)} decoder layers")

        ## tf.print("\nInput encoder\n", x_encoder[0][:3][:6])
        ## tf.print(x_encoder.shape)
        ## tf.print("Input decoder\n", x_decoder[0][:3][:6])
        ## tf.print(x_decoder.shape)

        if self.num_encoder_layers > 0:
            seq_len = tf.shape(x_encoder)[1]

            if self.input_vocab_size is None:
                if x_encoder.ndim != 3:
                    raise ValueError("There is no embadings in inputs. Make sure your data shape looks like (batch_size, seq_len, d_model)")
            else:
                if x_encoder.ndim != 2:
                    raise ValueError("Wrong inputs. Make sure your data shape looks like (batch_size, seq_len)")
                x_encoder = self.encoder_embedding(x_encoder)

            x_encoder *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

            ## tf.print("Before pos encoding\n", x_encoder[0][:3][:6])
            ## tf.print(x_encoder.shape)

            x_encoder += self.pos_encoding[:, :seq_len, :]
   
            x_encoder = self.dropout(x_encoder, training=training)

            ## tf.print("Pos encoding \n", self.pos_encoding[:3][:6])
            ## tf.print("After pos encoding \n", x_encoder[0][:3][:6])
            ## tf.print(x_encoder.shape, "\n")

            for i, layer in enumerate(self.encoder):
                x_encoder = layer(x_encoder, training=training, padding_mask= encoder_padding_mask)
                ## tf.print(f"Encoder layer {i} output \n", x_encoder[0][:3][:6])
                # print(x_encoder.shape)
        
        if self.num_decoder_layers > 0:
            seq_len = tf.shape(x_decoder)[1]

            if self.target_vocab_size is None:
                if x_decoder.ndim != 3:
                    raise ValueError("There is no embadings in inputs. Make sure your data shape looks like (batch_size, seq_len, d_model)")

            else:
                if x_decoder.ndim != 2:
                    raise ValueError("Wrong inputs. Make sure your data shape looks like (batch_size, seq_len)")
                x_decoder = self.decoder_embedding(x_decoder)
              
            x_decoder *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x_decoder += self.pos_encoding[:, :seq_len, :]

            x_decoder = self.dropout(x_decoder, training=training)

            ## tf.print("\nAfter pos encoding min/max:", tf.reduce_min(x_decoder), tf.reduce_max(x_decoder))

            for i, layer in enumerate(self.decoder):
                x_decoder = layer(x_decoder, x_encoder, training=training, attention_mask= attention_mask, padding_mask= decoder_padding_mask)
                ## tf.print(f"Decoder layer {i} output min/max:", tf.reduce_min(x_decoder), tf.reduce_max(x_decoder))
        else:
            x_decoder = x_encoder

        output = self.linear(x_decoder)

        # with open("output.txt", "a") as out:
        #     for batch in output:
        #         string = ""
        #         for el in batch:
        #             string += str(np.argmax(el))
        #         out.write(string + "\n")
        #     out.write("\n")

        ## tf.print("\nFinal output \n", output[0][:3][:6])
        ## tf.print("Output shape:\n", tf.shape(output))
        
        # tf.debugging.check_numerics(output, "Output contains NaN or Inf")
        ## tf.print("TYPE RETURN:", type(output), output is None)

        return output


    def __str__(self):
        return f"Name: {self.name}\nEncoding: \n\tnumber: {self.num_encoder_layers}\nDecoding: \n\tnumber:{self.num_decoder_layers}\nOutput_size: {output_vocab_size}"
    

if __name__ == "__main__":

    num_encoders = 2
    num_decoders = 0
    d_model = 296
    num_heads = 8
    d_ff = 128
    output_vocab_size = 2
    seq_len = 40

    BATCH_SIZE = 32
    EPOCHS = 5  

    attention_mask = [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1]
    ]

    # padding_mask = [1, 1, 1, 1, 1, 0, 0]
    # padding_mask = np.array(padding_mask)    
    # padding_mask  = padding_mask[:, None, None, :]

    transformer  = Transformer(
        num_encoders,
        num_decoders, 
        d_model, 
        num_heads, 
        d_ff, 
        output_vocab_size, 
        seq_len
        )
    
    transformer.compile(
    optimizer= 'adam', 
    loss= keras.losses.CategoricalCrossentropy(from_logits=True), 
    metrics= ['accuracy'],
    weighted_metrics= ['accuracy']
    )
    
    transformer.name = "Test_model"

    tf.config.run_functions_eagerly(True)

    x_train = np.ones((175, seq_len, d_model))
    x_test = np.ones((59, seq_len, d_model))
    y_train = np.zeros((175, seq_len, output_vocab_size))
    y_test = np.zeros((59, seq_len, output_vocab_size))

    for i, val in enumerate(y_train):
        for j, el in enumerate(val):
            y_train[i][j][j % output_vocab_size] = 1

    for i, val in enumerate(y_test):
        for j, el in enumerate(val):
            y_test[i][j][j % output_vocab_size] = 1
    
    input = {
        "encoder_input": x_train,
        "decoder_input": np.zeros((len(x_train), seq_len, d_model)),
        # "attention_mask": None,
        # "encoder_padding_mask": padding_mask,
        # "decoder_padding_mask": None
    }

    dataset = tf.data.Dataset.from_tensor_slices((
        input,
        y_train
    ))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    validation_input = {
        "encoder_input": x_test,
        "decoder_input": np.zeros((len(x_test), seq_len, d_model)),
        # "attention_mask": None,
        # "encoder_padding_mask": padding_mask,
        # "decoder_padding_mask": None
    }

    validation_dataset = tf.data.Dataset.from_tensor_slices((
        validation_input,
        y_test
    ))
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    history = transformer.fit(
        dataset,
        epochs= EPOCHS,
        validation_data= validation_dataset
        )

