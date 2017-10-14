#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging
import csv
import jieba
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#sentences = word2vec.Text8Corpus(u"finalResult.txt")  # 加载语料
#model = word2vec.Word2Vec(sentences, size=20,min_count=1)
#y1 = model.similarity(u"北京", u"城市")
#print(u"【北京】和【城市】的相似度为：", y1)

#load and train dataSet and save train model
def load_Train(trainText):
    sentences = word2vec.Text8Corpus(trainText)
    model = word2vec.Word2Vec(sentences, size=20,min_count=1)
    model.save(u"word2vec.model")
    model.save_word2vec_format(u"word2vec.vector",binary=False)

def load_Train2(trainText):
    sentences = word2vec.LineSentence(trainText)
    model = word2vec.Word2Vec(sentences, size=200,min_count=1)
    model.save("word2vec.model")
    model.save_word2vec_format("word2vec.vector",binary=False)

def load_model():
    model = word2vec.Word2Vec.load_word2vec_format("word2vec.vector",binary=False);
    return model

def getSimilar(model,text):
    result = model.most_similar(text,topn=3)
    return result

def load_excell_file(fileName,colIndex1,colIndex2):
    csv_reader = csv.reader(open(fileName))
    f1 = open("finalResult.txt",'a');
    for row in csv_reader:
        seg_list = jieba.cut(row[colIndex1]);
        verbs = " ".join(seg_list).split(" ")
        for verb in verbs:
            verb = verb.replace("1","");
            verb = verb.replace("2","");
            verb = verb.replace("3","");
            verb = verb.replace("4","");
            verb = verb.replace("5","");
            verb = verb.replace("6","");
            verb = verb.replace("7","");
            verb = verb.replace("8","");
            verb = verb.replace("9","");
            verb = verb.replace("0","");
            verb = verb.replace("|","");
            verb = verb.replace("$","");
            verb = verb.replace(":","");
            verb = verb.replace(";","");
            verb = verb.replace("-","");
            verb = verb.replace("&","");
            verb = verb.replace(".","");
            verb = verb.replace(",","");
            verb = verb.replace("  "," ");
            verb = verb.replace("   "," ");
            f1.write(" "+ verb);
        seg_list1 = jieba.cut(row[colIndex2]);
        verbs2 = "".join(seg_list1).split(" ")
        for verb in verbs2:
            verb = verb.replace("1","");
            verb = verb.replace("2","");
            verb = verb.replace("3","");
            verb = verb.replace("4","");
            verb = verb.replace("5","");
            verb = verb.replace("6","");
            verb = verb.replace("7","");
            verb = verb.replace("8","");
            verb = verb.replace("9","");
            verb = verb.replace("0","");
            verb = verb.replace("|","");
            verb = verb.replace("$","");
            verb = verb.replace(":","");
            verb = verb.replace(";","");
            verb = verb.replace("-","");
            verb = verb.replace("&","");
            verb = verb.replace(".","");
            verb = verb.replace(",","");
            verb = verb.replace("  "," ");
            verb = verb.replace("   "," ");
            f1.write(" "+ verb);
        f1.write("\n");
    f1.close();

def getTrainNameList(fileName,colIndex,namelist):
    csv_reader = csv.reader(open(fileName,encoding='utf-8'))
    for row in csv_reader:
        namelist.append(row[colIndex]);
    return namelist;

def loadNameList():
    namelist = [];
    namelist = getTrainNameList("北京市文物商店.csv",1,namelist);
    namelist = getTrainNameList("超市.csv",5,namelist);
    namelist = getTrainNameList("星级饭店.csv",0,namelist);
    namelist = getTrainNameList("高校.csv",1,namelist);
    namelist = getTrainNameList("家政服务.csv",5,namelist);
    return namelist;
    
def load_file():
    load_excell_file("北京市文物商店.csv",1)
    load_excell_file("超市.csv",5)
    load_excell_file("星级饭店.csv",0)
    load_excell_file("高校.csv",1)
    load_excell_file("家政服务.csv",5)

def cut_namelist(namelist,model):
    total_train_set = [];
    for name in namelist:
        namevalue = np.array([0,0,0]);
        seg_list = jieba.cut(name)
        vocablist = ",".join(seg_list).split(",")
        for verb in vocablist:
            verb = verb.replace("（","");
            verb = verb.replace("）","");
            verb = verb.replace("(","");
            verb = verb.replace(")","");
            verb = verb.replace("：","");
            verb = verb.replace("、","");
            verb = verb.replace("\n","");
            verb = verb.replace("?","");
            verb = verb.replace("-","");
            verb = verb.replace("/","");
            verb = verb.replace("\\","");
            verb = verb.replace(" ","");
            verb = verb.replace(".","");
            verb = verb.replace("·","");
            verb = verb.replace("▲","");
            verb = verb.replace("\u3000","");
            vector = [];
            if verb!="":
                result = getSimilar(model,verb);
                vector.append(result[0][1])
                vector.append(result[1][1])
                vector.append(result[2][1])
                namevalue = namevalue + np.array(vector)
        total_train_set.append(namevalue);
    return total_train_set;

def test_main():
    model = load_model();
    namelist = load_file();
    print(namelist);
    #total_train_set = cut_namelist(namelist,model);
    #return total_train_set;
