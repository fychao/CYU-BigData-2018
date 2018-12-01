import flask
import re
import os
import random
import dill
import json
import jieba
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request

# 載入模型檔
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
    mstr = "AUG 使用的模式"

    mstr = request.args.get("mstr")
    print(mstr)

    input_vector = tfidf.transform([token(mstr)])
    return "%s"%kmeans.predict(input_vector)




def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    return zip(feature_vals, score_vals)

def getJobFeature(fn_num=5):
	bdir = "./jobs/"
	# 以下為 ucan 文本集合使用
	#bdir = "./ucan/"
	pick_files = random.sample(os.listdir(bdir), fn_num)

	cnts = [ json.loads(open(bdir+pick_file, 'r', encoding='utf8').read())['job_desc'] for pick_file in pick_files]

	# 以下為 ucan 文本集合使用
	#cnts = [ open(bdir+pick_file, 'r', encoding='utf8').read() for pick_file in pick_files]

	ans = tfidf.transform([token(cnt) for cnt in cnts])

	feature_names=tfidf.get_feature_names()

	#sort the tf-idf vectors by descending order of scores
	sorted_items=sort_coo(ans.tocoo())

	#extract only the top n; n here is 10
	keywords=extract_topn_from_vector(feature_names, sorted_items, 100)
	 
	# # now print the results
	# print("\n===Keywords===")
	# for (w, s) in keywords:
	#    print("%s=%.2f"%(w, s))
	return list(set([ w for (w, s) in keywords]))



@app.route("/questions/")
def questions():
    
    num = request.args.get("num")

    return_q = []
    for idx in range(int(num)):
        wrds = getJobFeature(fn_num=2)[:10]
        return_q.append((idx, wrds))

    return_q = dict(return_q)
    #input_vector = tfidf.transform([token(mstr)])
    return "我要吐 %s 題數, 抓到的字詞：%s"%(num, return_q)


if __name__ == '__main__':
    app.run(debug=True)