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

mstr = ""
input_vector = tfidf.transform([token(mstr)])
kmeans.predict(input_vector)

app = Flask(__name__)

@app.route("/")
def hello():
    mstr = ""

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
filter_wrds = set(["如何", "各項", "事宜", "須長", "合者", "法相", "公斤", "再過", "處理", "電洽", "客訂", "親洽", "時需", "依顧", 
    "具備", "印前", "相關", "有意", "師或", "進行", "整進", "可談", "需會", "意者", "仁武", "不必", "亦可", "料及", "開立", "單備", 
    "缺為", "面談", "面試", "使用", "時間", "每個", "與業", "錄取", "天敵", "並能", "計前", "上下", "交付", "交辨", "一次", "通協",
    "估驗", "造廠", "手中", "依勞", "一般" ,"飲及", "有時", "新村", "工作", "其他", "制若", "適者", "要會", "理及", "超過", "開手",
    "填寫", "具有", "雅區", "優佳", "享有", "應徵", "無性", "最好", "意外", "送及", "部份", "增設", "履歷", "打擾", "經驗", "小姐",
    "元起", "澄清", "事項", "洽孫", "先電", "週五", "由佳", "級制", "符合", "累積", "先生", "章服", "接受", "簽名", "部門", "需具",
    "可依", "科系", "正本", "蓋章", "謝絕", "飲場", "從事", "人員", "以及", "本部", "有空", "先郵", "基準", "本職", "配合", "給付",
    "以內", "總館", "地點", "以下", "減輕", "提供", "專業", "週四", "公休", "有染", "小紋", "修水", "技術", "完成", "規定", "基本",
    "內容", "困擾", "能勇", "基法", "代表", "以上", "態度", "加入", "官方", "高職", "將顧", "根據", "收取", "計開", "一個", "事情",
    "有責", "任感", "請洽", "大潤", "優先", "可授", "開設", "依照", "謝謝", "需求", "備有", "三廠", "依據", "參考", "一切", "勿試",
    "付者", "先寄", "凍車", "八天", "附加", "及點", "擅長", "前往", "內業", "每餐", "各類", "及辦", "若上", "另有", "肧等", "磨毛",
    "誠徵", "交辦", "需要", "如需", "直接", "倍率", "科佳", "方式", "尤佳", "以北", "寒軒", "需輪", "此為", "過的", "最大", "需要",
    "讀生", "合計", "運面", "元至", "濾光", "扣款", "各戶", "粗化", "日常", "投遞", "臨至", "可至", "成為", "特定", "並執", "不得",
    "上班", "取得", "部定", "時段", "不用", "現有", "進出", "鐵牌", "自調", "須有", "八日", "負責", "例如", "競品", "三個", "至多",
    "需進", "登打", "除濕及", "應徵請", "地區備", "具服務", "心強", "語佳", "電洽莊", "小中", "筒中", "品換", "列及", "先考", "需長",
    "須視店", "發津", "儲作業", "進行進", "職人員", "戶書信", "極佳", "需願意", "關之產品", "推廣機", "處理國", "李小姐", "之業務",
    "並做", "除了", "須具護理", "術後及", "獎金及", "事代書", "服務及", "須具", "中班津", "以利", "必要", "此條件", "聘人員", "廠為",
    "年級", "自備", "並負責", "貓之作業", "需自備", "表電洽", "內事務", "空調場", "安裝熱", "無誠勿試", "課作業員", "即可", "再享",
    "當日", "履歷表", "具相關", "或大廈", "所有人", "之疑問", "負責為", "遞給", "並給予", "並處理", "樓下", "條件", "小時", "等等",
    "熟稔", "具親", "非常", "性質", "多采", "一份", "權東路", "職務後轉", "早晚", "將菜", "交顧客", "有生", "產設備", "人系統",
    "原安裝", "統圖", "之神經", "之經驗", "起聘", "貨狀況", "庫作業", "表至", "電連", "元統", "旅遊及", "一段", "包級及", "先電洽",
    "其品質", "各種", "職業類", "至桃園", "分派", "標作業", "需久", "度高", "觀念", "類檢", "隨談", "解聘", "機維修", "人大",
    "行軟", "關部門", "林小姐", "需輪班輪", "及其", "包及", "具責任", "公室", "烘布", "擺盤備", "機點", "此職", "並達成", "缺請",
    "抱負", "機門市", "父學習", "適會", "與業務", "設及", "須具備", "台維修", "鄭總務", "續會", "貨協助", "週一至", "數字敏", "中小",
    "通協調", "及人", "使人員", "林經理", "圖者", "李先生", "人士", "另行通知", "費另計", "寄至", "下列", "歡迎", "一餐", "植針",
    "設計及", "智障", "浪費", "是否", "擁有", "工資遣", "具責", "性佳", "職意", "機及", "需具備", "需住", "環控溫", "表交至",
    "警衛待", "及點貨", "學類", "收合者", "薪資", "通知", "名額", "身權法", "貨經驗", "並且", "並參考", "主溝通", "處理店", "單諮詢",
    "負責餐", "必電約", "極主動", "電洽施", "竊盜", "服員", "事產品", "需有", "人員資格", "及產品", "日止", "擇優", "候補", "保護處",
    "需領", "維護及", "整授", "應徵者", "貨檢驗", "並執行", "區鎮", "協尋與", "及愛心", "轉狀況", "或具", "另外", "獎懲", "搬運及",
    "之下", "經營績", "需懂", "有數", "運需", "人員入", "審資", "王先生", "星期一", "人佳", "四路", "鐘車程", "區明", "機設備", 
    "國定", "衛材倉", "優惠", "中間", "及安裝", "應無塵室", "具產品", "優先面", "請務必", "及維護", "完整", "公司", "能力", "基礎",
    "一律", "本校", "過程", "做事", "依職業", "之安裝", "力作", "歷合者", "將進", "傳真履", "之特性", "一件", "對產品", "至少",
    "我們", "產產品", "關生", "初期", "栗田集", "幫助", "之書圖", "及室", "清潔及", "單處理", "真至", "號或傳"])

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
    return ", ".join(list(set([ w for (w, s) in keywords if not w in filter_wrds and len(w) > 1]))[:10])



@app.route("/questions/")
def questions():
    
    num = request.args.get("num")

    return_q = []
    for idx in range(int(num)):
        wrds = getJobFeature(fn_num=2)
        return_q.append((idx, wrds))

    res = dict()
    res['qestions'] = dict(return_q)
    res['uuid'] = "%s"%uuid.uuid4()
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
    topn_jobs = sorted(dict(like_jobs).items(), key=operator.itemgetter(1), reverse=True)[:20]
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