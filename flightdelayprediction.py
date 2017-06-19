# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:27:43 2017

@author: schaa
"""

import pickle
import datetime as dt 
from sklearn.linear_model import LogisticRegression
from flask import Flask,render_template,request,redirect
import pandas as pd
import os
import numpy as np

classes=np.loadtxt('modclasses.txt', delimiter=',')
coef=np.loadtxt('modcoef.txt', delimiter=',')
coef=coef.reshape((1,coef.size))
intercept=np.loadtxt('modintercept.txt', delimiter=',')
testx=np.loadtxt('testx.txt', delimiter=',')
clf2=LogisticRegression(class_weight=None, verbose=5,penalty= 'l2')
clf2.classes_, clf2.coef_, clf2.intercept_ =classes, coef, intercept


clf=pickle.load( open( "logregmod.p", "rb" ) )
origins=pickle.load( open( "origins.p", "rb" ) )
destinations=pickle.load( open( "destinations.p", "rb" ) )

dic_carrier={'DL':'Delta (DL)','B6':'JetBlue (B6)','AA':'American (AA)',
             'AS':'Alaska (AS)','F9':'Frontier (F9)','HA':'Hawaiian (HA)',
             'NK':'Spirit (NK)','UA':'United (UA)','VX':'Virgin America (VX)',
             'WN':'Southwest (WN)','EV':'Atlantic Southeast (EV)',
             'OO':'SkyWest (OO)'}

dic_month={'1':'Jan','2':'Feb','3':'Mar','4':'Apr','5':'May','6':'Jun',
           '7':'Jul','8':'Aug','9':'Sep','10':'Oct','11':'Nov','12':'Dec'}

dic_day={str(i): str(i).zfill(2) for i in range(1,32)}
dic_origin={i: i for i in origins}
dic_destination={i: i for i in destinations}
dic_hour={str(i): str(i).zfill(2) for i in range(0,24)}
dic_minute={str(i): str(i).zfill(2) for i in range(0,60)}

tr=0.17
carrier = 'AA'
year = 0
month = 1
day = 1
origin = 'ABE'
destination = 'ABI'
hour = '0'
minute = '0'


def getdummies(name,options, eingabe,auszugebendes ):
    for o in options:
        auszugebendes.loc[1,name+'_'+ o]= 1 if o==eingabe else 0
    return auszugebendes


        
    
    


def mkopt(d,l,sel):
    s=''
    for i in l:
        if i!=sel:
            s+='<option value="'+i+'">'+d[i]+'</option>'
        else:
            s+='<option value="'+i+'" selected="selected">'+d[i]+'</option>'
    return s

app = Flask(__name__)
answer=''
@app.route('/',methods=['GET'])
def select():
    global year
    now=dt.datetime.now()
    dic_year={str(i): str(i) for i in range(now.year-5,now.year+2)}
    if int(year)<=0: year=str(now.year)
    return render_template('flightselect.html',opt_carrier=mkopt(dic_carrier,sorted(list(dic_carrier)),carrier),
                                               opt_year=mkopt(dic_year,sorted(list(dic_year),key=lambda x: int(x),reverse=True),year),
                                               opt_month=mkopt(dic_month,sorted(list(dic_month),key=lambda x: int(x)),month),
                                               opt_day=mkopt(dic_day,sorted(list(dic_day),key=lambda x: int(x)),day),
                                               opt_origin=mkopt(dic_origin,sorted(list(dic_origin)),origin),
                                               opt_destination=mkopt(dic_destination,sorted(list(dic_destination)),destination),
                                               opt_hour=mkopt(dic_hour,sorted(list(dic_hour),key=lambda x: int(x)),hour),
                                               opt_minute=mkopt(dic_minute,sorted(list(dic_minute),key=lambda x: int(x)),minute),
                                               answer=answer)


#clf = pickle.load( open( "logregmod.p", "rb" ) )
@app.route('/predict',methods=['POST'])
def predict():
    global answer,tr,carrier,year,month,day,origin,destination,hour,minute
    carrier = request.form['carrier']
    year = request.form['year']
    month = request.form['month']
    day = request.form['day']
    origin = request.form['origin']
    destination = request.form['destination']
    hour = request.form['hour']
    minute = request.form['minute']
    indata=pd.DataFrame()
    getdummies('CARRIER',list(dic_carrier), carrier, indata)
    getdummies('MONTH',list(dic_month), month, indata)
    getdummies('DAY_OF_MONTH',list(dic_day), day, indata)
    getdummies('DEST',list(dic_destination), destination, indata)
    getdummies('ORIGIN',list(dic_origin), origin, indata)
    getdummies('CRS_DEP_TIME',list(dic_hour), hour, indata)
    weekday=dt.datetime(int(year),int(month),int(day)).weekday()+1
    getdummies('DAY_OF_WEEK',[str(i) for i in range(1,8)], str(weekday), indata)
    indata=indata.reindex_axis(sorted(indata.columns), axis=1)
    indata.to_csv('static/inputdataloc.csv')
    indata2=pd.read_csv('static/inputdataloc.csv',index_col=0)
    pre = clf.predict(indata)
    pro = clf.predict_proba(indata)
    print('first pred', clf.predict_proba(indata))
    pre2 = clf2.predict_proba(indata)
    print('second pred', clf2.predict_proba(indata2))
    prstr='ontime' if pro[0,1]<tr else 'delayed'
    flightstr= origin+' to '+ destination+ ' on ' + month + '/'+ day+'/' +\
    year +   ' at ' + str(hour).zfill(2)+ ':' + str(minute).zfill(2) 
    answer='<p style="background-color:#d1efef;padding: 10%" >The flight from '+flightstr +\
        ' is predicted <br><font size=10>'+ prstr+'</font></p>'
    
    return redirect('/')
    
    
    
    
if __name__ == "__main__":
    #app.run(debug=False)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    
