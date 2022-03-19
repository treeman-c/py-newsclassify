import itertools
import os.path
import re
import jieba
import joblib
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier as knn


import NewsSpider as ns
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer  # TF-IDF向量转换类


def TrainModel(dt):
    dt = ns.DelStopword(dt)
    type = dt['文本类别']
    keyword = dt['keyword']
    count = CountVectorizer()
    x, x_test, y, y_test = train_test_split(keyword, type, test_size=0.2, random_state=0)
    x = count.fit_transform(x)
    tfidf_transformer = TfidfTransformer()
    x = tfidf_transformer.fit_transform(x)
    model = MultinomialNB().fit(x, y)
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./model/newsmodel.pkl'):
        f = open('./model/newsmodel.pkl', 'wb')
        f.close()
    if not os.path.exists('./model/vocab.pkl'):
        f = open('./model/vocab.pkl', 'wb')
        f.close()
    joblib.dump(model, './model/newsmodel.pkl')
    joblib.dump(count.vocabulary_, 'model/vocab.pkl')
    print("分类准确率:", accuracy_score(y_test, model.predict(count.transform(x_test))) * 100)
    print("分类评估报告如下:\n")
    print(classification_report(y_test, model.predict(count.transform(x_test))))


# 测试文本
def testmodel(text):
    delOrther = re.compile(r"[^a-zA-Z0-9\u4E00-\u9FA5]")
    text = delOrther.sub('', text)  # 去除非汉字字母数字的符号
    stopworlds = ns.GetStopword()
    text = " ".join([w for w in list(jieba.cut(text)) if w not in stopworlds])
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./model/newsmodel.pkl'):
        f = open('./model/newsmodel.pkl', 'wb')
        f.close()
    model = joblib.load('./model/newsmodel.pkl')
    vob = joblib.load('./model/vocab.pkl')
    count = CountVectorizer(ngram_range=(1, 4), min_df=1, vocabulary=vob)
    count._validate_vocabulary()
    text = [text]
    print(model.predict(count.transform(text))[0])
    return model.predict(count.transform(text))[0]


##svm支持向量机相关代码##
def train_svm_model(dt):
    dt = ns.DelStopword(dt)
    type = dt['文本类别']
    keyword = dt['keyword']
    count = CountVectorizer()
    x, x_test, y, y_test = train_test_split(keyword, type, test_size=0.2, random_state=0)
    x = count.fit_transform(x)
    tfidf_transformer = TfidfTransformer()
    x = tfidf_transformer.fit_transform(x)
    constructor = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=0))
    model = constructor.fit(x, y)
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./model/svm_model.pkl'):
        f = open('./model/svm_model.pkl', 'wb')
        f.close()
    if not os.path.exists('./model/svm_vocab.pkl'):
        f = open('./model/svm_vocab.pkl', 'wb')
        f.close()
    joblib.dump(model, './model/svm_model.pkl')
    joblib.dump(count.vocabulary_, 'model/svm_vocab.pkl')
    print("分类准确率:", accuracy_score(y_test, model.predict(count.transform(x_test))) * 100)
    print("分类评估报告如下:\n")
    print(classification_report(y_test, model.predict(count.transform(x_test))))


def test_svm_model(text):
    delOrther = re.compile(r"[^a-zA-Z0-9\u4E00-\u9FA5]")
    text = delOrther.sub('', text)  # 去除非汉字字母数字的符号
    stopworlds = ns.GetStopword()
    text = " ".join([w for w in list(jieba.cut(text)) if w not in stopworlds])
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./model/svm_model.pkl'):
        f = open('./model/svm_model.pkl', 'wb')
        f.close()
    model = joblib.load('./model/svm_model.pkl')
    vob = joblib.load('./model/svm_vocab.pkl')
    count = CountVectorizer(ngram_range=(1, 4), min_df=1, vocabulary=vob)
    count._validate_vocabulary()
    text = [text]
    print(model.predict(count.transform(text))[0])
    return model.predict(count.transform(text))[0]


##k最近邻knn相关代码##
def train_knn_model(dt):
    dt = ns.DelStopword(dt)
    type = dt['文本类别']
    keyword = dt['keyword']
    count = CountVectorizer()
    x, x_test, y, y_test = train_test_split(keyword, type, test_size=0.2, random_state=0)
    x = count.fit_transform(x)
    tfidf_transformer = TfidfTransformer()
    x = tfidf_transformer.fit_transform(x)
    constructor = knn(n_neighbors=5, weights="distance", algorithm="auto", n_jobs=1)
    model = constructor.fit(x, y)
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./model/knn_model.pkl'):
        f = open('./model/knn_model.pkl', 'wb')
        f.close()
    if not os.path.exists('./model/knn_vocab.pkl'):
        f = open('./model/knn_vocab.pkl', 'wb')
        f.close()
    joblib.dump(model, './model/knn_model.pkl')
    joblib.dump(count.vocabulary_, 'model/knn_vocab.pkl')
    print("分类准确率:", accuracy_score(y_test, model.predict(count.transform(x_test))) * 100)
    print("分类评估报告如下:\n")
    print(classification_report(y_test, model.predict(count.transform(x_test))))


def test_knn_model(text):
    delOrther = re.compile(r"[^a-zA-Z0-9\u4E00-\u9FA5]")
    text = delOrther.sub('', text)  # 去除非汉字字母数字的符号
    stopworlds = ns.GetStopword()
    text = " ".join([w for w in list(jieba.cut(text)) if w not in stopworlds])
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./model/knn_model.pkl'):
        f = open('./model/knn_model.pkl', 'wb')
        f.close()
    model = joblib.load('./model/knn_model.pkl')
    vob = joblib.load('./model/knn_vocab.pkl')
    count = CountVectorizer(ngram_range=(1, 4), min_df=1, vocabulary=vob)
    count._validate_vocabulary()
    text = [text]
    print(model.predict(count.transform(text))[0])
    return model.predict(count.transform(text))[0]


# if __name__ == '__main__':
def init():
    ns.GetData()
    data = pd.read_csv('newdata.csv', encoding='utf-8', engine='python')
    TrainModel(data)
    testmodel('当地时间2022年2月12日，克里米亚塞瓦斯托波尔巴拉克拉瓦， 巴拉克拉瓦地下博物馆中的展品。巴拉克拉瓦地下博物馆是军事历史防御工事博物馆的一个分馆，这里原先是一个潜艇基地。')
    train_svm_model(data)
    test_svm_model('当地时间2022年2月12日，克里米亚塞瓦斯托波尔巴拉克拉瓦， 巴拉克拉瓦地下博物馆中的展品。巴拉克拉瓦地下博物馆是军事历史防御工事博物馆的一个分馆，这里原先是一个潜艇基地。')
    train_knn_model(data)
    test_knn_model('当地时间2022年2月12日，克里米亚塞瓦斯托波尔巴拉克拉瓦， 巴拉克拉瓦地下博物馆中的展品。巴拉克拉瓦地下博物馆是军事历史防御工事博物馆的一个分馆，这里原先是一个潜艇基地。')
    # return data
