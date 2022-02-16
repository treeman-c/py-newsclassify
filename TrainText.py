import itertools
import os.path
import re

import jieba
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import NewsSpider as ns
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer  # TF-IDF向量转换类


def TrainModel(dt):
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
    plot_confusion_matrix(y_test, model.predict(count.transform(x_test)))


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


# 绘制混淆矩阵函数
def plot_confusion_matrix(y_test, y_pred,
                          normalize=False,
                          cmap=plt.cm.Blues):
    classes = ['军事', '国内', '国际', '科技', '航空']
    cm = confusion_matrix(y_test, y_pred)
    title = "分类准确率:{:.2f}%".format(accuracy_score(y_test, y_pred) * 100)
    print("分类评估报告如下:\n")
    print(classification_report(y_test, y_pred))
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()


if __name__ == '__main__':
# def init():
    data = pd.read_csv('newdata.csv', encoding='utf-8', engine='python')
    data = ns.DelStopword(data)
    ns.GetData()
    TrainModel(data)
    testmodel('当地时间2022年2月12日，克里米亚塞瓦斯托波尔巴拉克拉瓦， 巴拉克拉瓦地下博物馆中的展品。巴拉克拉瓦地下博物馆是军事历史防御工事博物馆的一个分馆，这里原先是一个潜艇基地。')
    # return data
