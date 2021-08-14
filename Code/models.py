from hypers import *

import numpy
import keras
from keras.utils.np_utils import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, concatenate
from keras.layers import Dense, Input, Flatten, average,Lambda

from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers #keras2
from keras.utils import plot_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from keras.optimizers import *



class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

def AttentivePooling(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')
    user_vecs =Dropout(0.2)(vecs_input)
    user_att = Dense(200,activation='tanh')(user_vecs)
    user_att = keras.layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])
    model = Model(vecs_input,user_vec)
    return model

def ConDot():
    vec_input = keras.layers.Input(shape=(400*2,))
    vec1 = keras.layers.Lambda(lambda x:x[:,:400])(vec_input)
    vec2 = keras.layers.Lambda(lambda x:x[:,400:])(vec_input)
    score = keras.layers.Dot(axes=-1)([vec1,vec2])
    return Model(vec_input,score)

def get_doc_encoder(title_word_embedding_matrix,entity_emb_matrix):

    news_input = Input(shape=(35,),dtype='int32')
    
    
    sentence_input = keras.layers.Lambda(lambda x:x[:,:30])(news_input)
    title_word_embedding_layer = Embedding(title_word_embedding_matrix.shape[0], 300, weights=[title_word_embedding_matrix],trainable=True)
    word_vecs = title_word_embedding_layer(sentence_input)
    droped_vecs = Dropout(0.2)(word_vecs)
    word_rep = Attention(20,20)([droped_vecs]*3)
    droped_rep = Dropout(0.2)(word_rep)
    title_vec = AttentivePooling(30,400)(droped_rep)
    
    entity_input = keras.layers.Lambda(lambda x:x[:,30:])(news_input)
    entity_embedding_layer = Embedding(entity_emb_matrix.shape[0], 100, weights=[entity_emb_matrix],trainable=True)
    entity_vecs = entity_embedding_layer(entity_input)
    droped_vecs = Dropout(0.2)(entity_vecs)
    entity_rep = Attention(5,20)([droped_vecs]*3)
    droped_rep = Dropout(0.2)(entity_rep)
    entity_vec = AttentivePooling(5,100)(droped_rep)
    
    vec = keras.layers.Concatenate(axis=-1)([title_vec,entity_vec])
    vec = keras.layers.Dense(400)(vec)
    
    
    sentEncodert = Model(news_input, vec)
    return sentEncodert

class CategoryEmbLayer(Layer):
    
    def __init__(self,n, **kwargs):
        super(CategoryEmbLayer, self).__init__(**kwargs)
        self.n = n
        
     
    def build(self, input_shape):
        trainable = True
        if self.n>1:
            self.W = self.add_weight(name='W',
                                  shape=(self.n,400),
                                  initializer=keras.initializers.Constant(value=np.zeros((self.n,400))),
                                  trainable=trainable)
        else:
            self.W = self.add_weight(name='W',
                                  shape=(400,),
                                  initializer=keras.initializers.Constant(value=np.zeros((400,))),
                                  trainable=trainable)
            
    def call(self,x):
        return x+self.W
        
    def compute_output_shape(self, input_shape):
        return input_shape

class Weighter(Layer):
     
    def __init__(self, **kwargs):
        super(Weighter, self).__init__(**kwargs)
        

     
    def build(self, input_shape):
        trainable = False
        self.w1 = self.add_weight(name='w1',
                                  shape=(1,),
                                  initializer=keras.initializers.Constant(value=0.15/0.15),
                                  trainable=trainable)

        self.w2 = self.add_weight(name='w2',
                                  shape=(1,),
                                  initializer=keras.initializers.Constant(value=0.15/0.15),
                                  trainable=trainable)
        
        self.w3 = self.add_weight(name='w3',
                                  shape=(1,),
                                  initializer=keras.initializers.Constant(value=0.7/0.15),
                                  trainable=trainable)

        
        super(Weighter, self).build(input_shape)
        
        
    def call(self,x):

        return self.w1*x[0]+self.w2*x[1]+self.w3*x[2]
        
    def compute_output_shape(self, input_shape):

        return input_shape[0]

def HirUserEncoder(category_dict,subcategory_dict):
    
    AttTrainable = True
    
    clicked_title_input = Input(shape=(50,400,), dtype='float32')
    
    clicked_vert_input = Input(shape=(len(category_dict),50,), dtype='float32')
    clicked_vert_mask_input = Input(shape=(len(category_dict),), dtype='float32')
    
    clicked_subvert_input = Input(shape=(len(subcategory_dict),50,), dtype='float32')
    clicked_subvert_mask_input = Input(shape=(len(subcategory_dict),), dtype='float32')
    
    vert_subvert_mask_input = Input(shape=(len(category_dict),len(subcategory_dict)),dtype='float32')

    vert_num_input = Input(shape=(len(category_dict),),dtype='int32')
    subvert_num_input = Input(shape=(len(subcategory_dict),),dtype='int32')

    subvert_num_embedding_layer = Embedding(51, 128,trainable=True)
    subvert_num_scorer = Dense(1)


    vert_num_embedding_layer = subvert_num_embedding_layer #Embedding(51, 128,trainable=True)
    vert_num_scorer = subvert_num_scorer

    title_vecs = clicked_title_input
    
    trainable = True
    
    user_subvert_att = Dense(1,trainable=trainable,use_bias=False,kernel_initializer=keras.initializers.Constant(value=np.zeros((400,1))),)(title_vecs)

    user_subvert_att = keras.layers.Reshape((50,))(user_subvert_att)
    user_subvert_att = keras.layers.RepeatVector(len(subcategory_dict))(user_subvert_att)
    user_subvert_att = keras.layers.Lambda(lambda x:x[0]-100*(1-x[1]))([user_subvert_att,clicked_subvert_input])    
    user_subvert_att = keras.layers.Activation('softmax')(user_subvert_att) #(300,50)

    user_subvert_att = keras.layers.Lambda(lambda x:x[0]*x[1])([user_subvert_att,clicked_subvert_input]) #(300,400)
    user_subvert_rep = keras.layers.Dot(axes=[-1,-2])([user_subvert_att,title_vecs]) #（300,400)
    user_subvert_rep = CategoryEmbLayer(len(subcategory_dict))(user_subvert_rep)  #（300,400) 
    
    
    subvert_num_emb = subvert_num_embedding_layer(subvert_num_input)
    subvert_num_score = subvert_num_scorer(subvert_num_emb)
    subvert_num_score = Reshape((len(subcategory_dict),))(subvert_num_score) #(300,)   
    
    user_vert_att = Dense(1,trainable=trainable,use_bias=False,kernel_initializer=keras.initializers.Constant(value=np.zeros((400,1))))(user_subvert_rep)
    user_vert_att = Reshape((len(subcategory_dict),))(user_vert_att) #(300,)
    user_vert_att = Add()([user_vert_att,subvert_num_score]) #(300,)
    
    user_vert_att = RepeatVector(len(category_dict))(user_vert_att) #(18,300)
    user_vert_att = Lambda(lambda x:x[0]-100*(1-x[1]))([user_vert_att,vert_subvert_mask_input]) #(18,300)
    user_vert_att = Softmax()(user_vert_att)
    
    user_vert_rep = keras.layers.Dot(axes=[-1,-2])([user_vert_att,user_subvert_rep]) #(18,400)
    user_vert_rep = CategoryEmbLayer(len(category_dict))(user_vert_rep) #(18,400)

    user_global_att = Dense(1,trainable=trainable,use_bias=False,kernel_initializer=keras.initializers.Constant(value=np.zeros((400,1))))(user_vert_rep)
    user_global_att = Reshape((len(category_dict),))(user_global_att) #(18,)

    vert_num_emb = vert_num_embedding_layer(vert_num_input)
    vert_num_score = vert_num_scorer(vert_num_emb)
    vert_num_score = Reshape((len(category_dict),))(vert_num_score) #(18,1)   

    user_global_att = Add()([user_global_att,vert_num_score]) #(18,)
    user_global_att = Lambda(lambda x:x[0]-100*(1-x[1]))([user_global_att,clicked_vert_mask_input]) #(18,)
    user_global_att = Softmax()(user_global_att)
    
        
    user_global_rep = Dot(axes=[-1,-2])([user_global_att,user_vert_rep]) #(400,)
    
    return Model([clicked_title_input,clicked_vert_input,clicked_vert_mask_input,clicked_subvert_input,clicked_subvert_mask_input,vert_subvert_mask_input,vert_num_input,subvert_num_input],
                 [user_subvert_rep,user_vert_rep,user_global_rep])


def create_model(category_dict,subcategory_dict,title_word_embedding_matrix,entity_emb_matrix):
    MAX_LENGTH = 35    
    news_encoder = get_doc_encoder(title_word_embedding_matrix,entity_emb_matrix)

    user_encoder = HirUserEncoder(category_dict,subcategory_dict)
    
    clicked_title_input = Input(shape=(50,35,), dtype='int32')
    clicked_vert_input = Input(shape=(len(category_dict),50,), dtype='float32')
    clicked_vert_mask_input = Input(shape=(len(category_dict),), dtype='float32')
    clicked_subvert_input = Input(shape=(len(subcategory_dict),50,), dtype='float32')
    clicked_subvert_mask_input = Input(shape=(len(subcategory_dict),), dtype='float32')
    vert_subvert_mask_input = Input(shape=(len(category_dict),len(subcategory_dict)), dtype='float32')
    
    title_inputs = Input(shape=(1+npratio,35,),dtype='int32') 
    vert_inputs = Input(shape=(1+npratio,len(category_dict),),dtype='float32')  #(2,18)
    subvert_inputs = Input(shape=(1+npratio,len(subcategory_dict),),dtype='float32')  #(2,18)

    vert_num_input = Input(shape=(len(category_dict),),dtype='int32')
    subvert_num_input = Input(shape=(len(subcategory_dict),),dtype='int32')
    
    rw_vert_input = Input(shape=(1+npratio,),dtype='float32')
    rw_subvert_input = Input(shape=(1+npratio,),dtype='float32')

    clicked_title_vecs = TimeDistributed(news_encoder)(clicked_title_input)
    news_vecs = TimeDistributed(news_encoder)(title_inputs)
    
    news_vecs = Dropout(0.25)(news_vecs)
    clicked_title_vecs = Dropout(0.25)(clicked_title_vecs)

    user_subvert_rep,user_vert_rep,user_global_rep = user_encoder([clicked_title_vecs,clicked_vert_input,clicked_vert_mask_input,clicked_subvert_input,clicked_subvert_mask_input,vert_subvert_mask_input,vert_num_input,subvert_num_input])
    
    
    vs_user_vec = keras.layers.Dot(axes=(-1,-2))([vert_inputs,user_vert_rep]) #(batch_size,1+npratio,400)
    svs_user_vec = keras.layers.Dot(axes=(-1,-2))([subvert_inputs,user_subvert_rep]) #(batch_size,1+npratio,400)


    score1 = keras.layers.Dot(axes=-1)([news_vecs,user_global_rep])

    vs_vecs = keras.layers.Concatenate(axis=-1)([news_vecs,vs_user_vec])
    score2 = TimeDistributed(ConDot())(vs_vecs)
    score2 = keras.layers.Reshape((1+npratio,))(score2)
    
    svs_vecs = keras.layers.Concatenate(axis=-1)([news_vecs,svs_user_vec])
    score3 = TimeDistributed(ConDot())(svs_vecs)
    score3 = keras.layers.Reshape((1+npratio,))(score3)
    
    
    score2 = Multiply()([rw_vert_input,score2])
    score3 = Multiply()([rw_subvert_input,score3])

    rwer = Weighter()
    scores = rwer([score1,score2,score3])
    
    
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)     

    model = Model([title_inputs,vert_inputs,subvert_inputs,
                   clicked_title_input,clicked_vert_input,clicked_vert_mask_input,
                   clicked_subvert_input,clicked_subvert_mask_input,
                   vert_subvert_mask_input,vert_num_input,subvert_num_input,
                  rw_vert_input,rw_subvert_input],logits) # max prob_click_positive
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=Adam(lr=0.0001,amsgrad=True),
                  metrics=['acc'])

    
    return model,news_encoder,user_encoder,rwer