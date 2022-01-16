from hypers import *
from datetime import datetime
import time
import random
import numpy as np
import os
from nltk.tokenize import word_tokenize
import json

def trans2tsp(timestr):
    return int(time.mktime(datetime.strptime(timestr, '%m/%d/%Y %I:%M:%S %p').timetuple()))

def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)

def shuffle(pn,labeler,pos):
    index=np.arange(pn.shape[0])
    pn=pn[index]
    labeler=labeler[index]
    pos=pos[index]
    
    for i in range(pn.shape[0]):
        index=np.arange(npratio+1)
        pn[i,:]=pn[i,index]
        labeler[i,:]=labeler[i,index]
    return pn,labeler,pos

def read_news(path,filenames):
    news={}
    category=[]
    subcategory=[]
    news_index={}
    index=1
    word_dict={}
    word_index=1
    with open(os.path.join(path,filenames)) as f:
        lines=f.readlines()
    for line in lines:
        splited = line.strip('\n').split('\t')
        doc_id,vert,subvert,title= splited[0:4]
        news_index[doc_id]=index
        index+=1
        category.append(vert)
        subcategory.append(subvert)
        title = title.lower()
        title=word_tokenize(title)
        news[doc_id]=[vert,subvert,title]
        for word in title:
            word = word.lower()
            if not(word in word_dict):
                word_dict[word]=word_index
                word_index+=1
    category=list(set(category))
    subcategory=list(set(subcategory))
    category_dict={}
    index=1
    for c in category:
        category_dict[c]=index
        index+=1
    subcategory_dict={}
    index=1
    for c in subcategory:
        subcategory_dict[c]=index
        index+=1
    return news,news_index,category_dict,subcategory_dict,word_dict

def get_doc_input(news,news_index,category,subcategory,word_dict):
    news_num=len(news)+1
    news_title=np.zeros((news_num,MAX_SENTENCE),dtype='int32')
    news_vert=np.zeros((news_num,),dtype='int32')
    news_subvert=np.zeros((news_num,),dtype='int32')
    for key in news:    
        vert,subvert,title=news[key]
        doc_index=news_index[key]
        news_vert[doc_index]=category[vert]
        news_subvert[doc_index]=subcategory[subvert]
        for word_id in range(min(MAX_SENTENCE,len(title))):
            news_title[doc_index,word_id]=word_dict[title[word_id].lower()]
    return news_title,news_vert,news_subvert

def load_matrix(embedding_path,word_dict):
    embedding_matrix = np.zeros((len(word_dict)+1,300))
    have_word=[]
    with open(os.path.join(embedding_path,'glove.840B.300d.txt'),'rb') as f:
        while True:
            l=f.readline()
            if len(l)==0:
                break
            l=l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix[index]=np.array(tp)
                have_word.append(word)
    return embedding_matrix,have_word

def read_clickhistory(news_index,data_root_path,filename):
    
    lines = []
    userids = []
    with open(os.path.join(data_root_path,filename)) as f:
        lines = f.readlines()
        
    sessions = []
    for i in range(len(lines)):
        _,uid,eventime, click, imps = lines[i].strip().split('\t')
        if click == '':
            clikcs = []
        else:
            clikcs = click.split()
        true_click = []
        for click in clikcs:
            if not click in news_index:
                continue
            true_click.append(click)
        pos = []
        neg = []
        for imp in imps.split():
            docid, label = imp.split('-')
            if label == '1':
                pos.append(docid)
            else:
                neg.append(docid)
        sessions.append([true_click,pos,neg])
    return sessions

def parse_user(news_index,session):
    user_num = len(session)
    user={'click': np.zeros((user_num,MAX_ALL),dtype='int32'),}
    for user_id in range(len(session)):
        tclick = []
        click, pos, neg =session[user_id]
        for i in range(len(click)):
            tclick.append(news_index[click[i]])
        click = tclick

        if len(click) >MAX_ALL:
            click = click[-MAX_ALL:]
        else:
            click=[0]*(MAX_ALL-len(click)) + click
            
        user['click'][user_id] = np.array(click)
    return user

def get_train_input(news_index,session):
    sess_pos = []
    sess_neg = []
    user_id = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _, poss, negs=sess
        for i in range(len(poss)):
            pos = poss[i]
            neg=newsample(negs,npratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)
    sess_all = np.zeros((len(sess_pos),1+npratio),dtype='int32')
    label = np.zeros((len(sess_pos),1+npratio))
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id,0] = news_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id,index] = news_index[neg]
            index+=1
        label[sess_id,0]=1
    user_id = np.array(user_id, dtype='int32')
    
    return sess_all, user_id, label

def get_test_input(news_index,session):
    
    Impressions = []
    userid = []
    for sess_id in range(len(session)):
        _, poss, negs = session[sess_id]
        imp = {'labels':[],
                'docs':[]}
        userid.append(sess_id)
        for i in range(len(poss)):
            docid = news_index[poss[i]]
            imp['docs'].append(docid)
            imp['labels'].append(1)
        for i in range(len(negs)):
            docid = news_index[negs[i]]
            imp['docs'].append(docid)
            imp['labels'].append(0)
        Impressions.append(imp)
        
    userid = np.array(userid,dtype='int32')
    
    return Impressions, userid,

def load_news_entity(news_index,KG_root_path):
    with open(os.path.join(KG_root_path,'Release_Small_title.tsv')) as f:
        lines = f.readlines()
    
    EntityId2Index = {}
    ctt = 1
    
    news_entity = {}
    g = []
    for i in range(len(lines)):
        d = json.loads(lines[i].strip('\n'))
        docid = d['doc_id']
        if not docid in news_index:
            continue
        news_entity[docid] = []
        entities = d['entities']
        for j in range(len(entities)):
            e = entities[j]['Label']
            eid = entities[j]['WikidataId']
            if not eid in EntityId2Index:
                EntityId2Index[eid] = ctt
                ctt += 1
            news_entity[docid].append([e,eid,EntityId2Index[eid]])
    
    meta_news_entity = {}
    news_entity2 = {}
    
    
    news_entity_id = {}
    for nid in news_entity:
        news_entity_id[nid] = []
        for e in news_entity[nid]:
            news_entity_id[nid].append(e[-2])
        news_entity_id[nid] = set(news_entity_id[nid])
        
    
    for docid in news_entity:
        meta_news_entity[docid] = news_entity[docid]
        news_entity2[docid] = []
        for v in news_entity[docid]:
            news_entity2[docid].append(v[-1])
        news_entity2[docid] = list(set(news_entity2[docid]))[:5]
        news_entity2[docid] = news_entity2[docid] + [0]*(5-len(news_entity2[docid]))
        news_entity2[docid] = np.array(news_entity2[docid])
    
    news_entity_np = np.zeros((len(news_entity2)+1,5),dtype='int32')
    for nid in news_index:
        nix = news_index[nid]
        news_entity_np[nix] = news_entity2[nid]
        
    return news_entity_id,news_entity_np,EntityId2Index

def load_entity_embedding(KG_root_path,EntityId2Index):
    entity_emb = np.zeros((len(EntityId2Index)+1,100))
    import pickle
    with open(os.path.join(KG_root_path,'title_entity_emb.pkl'),'rb') as f:
        title_entity_emb = pickle.load(f)
    for eid in EntityId2Index:
        eix = EntityId2Index[eid]
        entity_emb[eix] = title_entity_emb[eid]
    return entity_emb