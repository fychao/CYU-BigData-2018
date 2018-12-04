import flask
import re
import os
import random
import dill
import json
import uuid
import jieba
import operator

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
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

# 過濾不想要的詞彙
filter_wrds = set(["如何", "各項"])

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
	return ", ".join(list(set([ w for (w, s) in keywords if not w in filter_wrds and len(w)==2]))[:10])



@app.route("/questions/")
def questions():
    
    num = request.args.get("num")

    return_q = []
    for idx in range(int(num)):
        wrds = getJobFeature(fn_num=2)
        return_q.append((idx, wrds))

    res = dict()
    res['qestions'] = dict(return_q)
    #res['uuid'] = "%s"%uuid.uuid4()
    res['num'] = int(num)

    resp = flask.Response(json.dumps(res), 
    	mimetype="application/json")
    resp.headers['Access-Control-Allow-Origin'] = '*' # 不安全
    return resp

@app.route("/results/", methods=['POST'])
def results():
    str_like = request.form['like']
    str_dislike = request.form['dislike']


    vec_like = tfidf.transform([token(str_like)])
    vec_dislike = tfidf.transform([token(str_dislike)])

    wanted_k = kmeans.predict(vec_like) 

    # 從分群的資料抓 20筆出來
    bdir = "./jobs/"
    wanted_jobs = []
    isStopGetJob = False
    while not isStopGetJob:
        pick_file = random.sample(os.listdir(bdir), 1)
        job = json.loads(open(bdir+pick_file[0], 'r', encoding='utf8').read())
        job_k = kmeans.predict( tfidf.transform([token(job['job_desc'])]) )
        if wanted_k[0] == job_k[0]:
        	wanted_jobs.append(job)

        if len(wanted_jobs) == 20:
        	isStopGetJob = True

    vec_wanted_job = [ tfidf.transform([token(job['job_desc'])]) for job in wanted_jobs]

    # 計算相似度模型
    like_jobs = []
    for idx in range(len(vec_wanted_job)):
    	like_jobs.append( (idx, cosine_similarity(vec_like[0], vec_wanted_job[idx])[0][0]) )

    # 取出最符合的10項工作
    topn_jobs = sorted(dict(like_jobs).items(), key=operator.itemgetter(1), reverse=True)[:10]
    like_job = [wanted_jobs[k] for (k, sim) in topn_jobs]

    # 將符合工作項目 再向量化
    vec_like_job = [ tfidf.transform([token(job['job_desc'])]) for job in like_job]
    print(vec_like_job)

    # 計算相似度模型
    unlike_jobs = []
    for idx in range(len(vec_like_job)):
    	unlike_jobs.append( (idx, cosine_similarity(vec_dislike[0], vec_like_job[idx])[0][0]) )

    # 找出前5個最不想要的
    topn_jobs_dislike = sorted(dict(unlike_jobs).items(), key=operator.itemgetter(1), reverse=True)[:5]
    dislike_job_id = set([ idx for (idx, value) in topn_jobs_dislike])

    # 把不想要的過濾掉
    final_proposed = "系統推薦：<ul>"
    for idx in range(len(like_job)):
    	if idx in dislike_job_id:
    		print ("dislike")
    	else:
    		final_proposed += "<li>%s</li>"%like_job[idx]['job_title']

    final_proposed +=  "</ul>"

    print (final_proposed)

    resp = flask.Response(json.dumps(final_proposed),
             mimetype="application/json")
    resp.headers['Access-Control-Allow-Origin'] = '*' # 不安全
    return resp

if __name__ == '__main__':
    app.run(debug=True)