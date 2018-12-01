import flask
import re
import dill
import jieba
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request

tfidf = dill.load(open("model_tfidf.dil", 'rb'))
kmeans = dill.load(open("model_kmeans.dil", 'rb'))

def token(txt):
    wrds = []
    for wrd in jieba.cut(txt):
        if re.match(u"[\u4e00-\u9fa5]+", wrd): #中文字
            wrds.append(wrd)
#         elif re.match("\d+", wrd): # 數字
#             wrds.append(wrd)
#         elif re.match("\w+", wrd): # 英文字
#             wrds.append(wrd)
        else:
            pass
    return wrds

mstr = "AUG 使用的模式"
input_vector = tfidf.transform([token(mstr)])
kmeans.predict(input_vector)

app = Flask(__name__)

@app.route("/")
def hello():
    mstr = request.args.get('mstr')
    input_vector = tfidf.transform([token(mstr)])
    return "%s"%kmeans.predict(input_vector)

if __name__ == '__main__':
    app.run(debug=True)