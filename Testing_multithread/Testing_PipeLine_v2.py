#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import obspy
import common as com
from Classifier import Classifier
from Predictor import Predictor
from threading import Thread
import time
import random

# Note that you can activate the median filter if the CapsPhase output is not smooth.
#database connection
servername = "localhost"
username = "root"
password = ""
database = "earthquake"


#mydb = mysql.connector.connect(host=servername, user=username, passwd=password, database=database)
#mycursor = mydb.cursor()
print('connection Successful')


class Prediction_thread(Thread):

    def __init__(self, data, model, id=0):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.id = id
        self.data=data
        self.model=model
        self.pred=0

    def run(self):
        pslice_proc = self.model.preproc(self.data)
        pred = self.model.predict(pslice_proc)
        self.pred = self.model.postproc(pred)
        print(f'In Thread:: {self.id}: {self.pred}')       
        

mode = 'stftcnn7'
classifier = Classifier(mode=mode)
classifier.model.summary()
mpred = Predictor('m1')
tpred = Predictor('t1') 
st_p = obspy.read('data\SeedTestTraces\*.mseed')
st_p[0].stats

info = com.get_info(st_p[0])
stt, endt = info['stt'], info['endt']
act_f = info['f']
window = 4 # sec
overlap = 5 # sec
stream_picks=[]

npts = st_p[0].stats.npts
stt = st_p[0].stats.starttime
picks_dict = {}
picks_dict[classifier.mode]={}
for phase in classifier.phases:
    picks_dict[classifier.mode][phase]={}
    picks_dict[classifier.mode][phase]['utc']=[]
    picks_dict[classifier.mode][phase]['value']=[]
    picks_dict[classifier.mode][phase]['pick']=[] 
    picks_dict[classifier.mode][phase]['out']=[] 
    picks_dict[classifier.mode][phase]['out_t'] =[]

print(f'P arrival at {stt+60} and S arrival at {endt-200}')
channel = ['HHE', 'HHN', 'HHZ']

st_new = st_p.select(channel='HHZ')
st_new = st_p[0].copy()

for count, tm in enumerate(np.arange(info['stt'], info['endt']-window, overlap)):
    st_slice = st_new.slice(tm, tm+window)
    batch_data = classifier.preproc(st_slice)
    
    prediction = classifier.get_picks(batch_data)
    pred_phase = classifier.phases[np.argmax(prediction)]
    #picks_dict, _, pred_phases = classifier.post_proc(prediction=prediction, batch_data=batch_data, count=count, tm=tm, stt=info['stt'], f_act=info['f'], picks_dict=picks_dict)
    #print("Pick_dict_P:",picks_dict['stftcnn5']['p']['utc'])
    #print('pred:', pred_phase)
    if pred_phase == 'p':
        print()
        mprediction_thread = Prediction_thread(id='Amplitude', data=st_slice, model=mpred)
        tprediction_thread = Prediction_thread(id='time',data=st_slice, model=tpred)
        mprediction_thread.start()
        tprediction_thread.start()
        mprediction_thread.join()
        tprediction_thread.join()
        print(f'P arrival predicted at {tm+2}')
        print(f'S wave with amplitude {mprediction_thread.pred:.02f} is expected after {tprediction_thread.pred/act_f:.02f} seconds from P arrival')
        pred_s = mprediction_thread.pred
        pred_t = tprediction_thread.pred
        
        #insert_query = "insert into p_pred (Pwave,PredS) values ('" + str(picks_dict[mode]['p']['value']) + "','" + str(pred_s) + "')"
        #print(insert_query)
        #mycursor = mydb.cursor()
        #mycursor.execute(insert_query)
        #mydb.commit()
        #print("Data Inserted in P table!")
        
    if pred_phase == 's':
        insert_query2 = "insert into detecteds (DetectedS) values ('" + str(picks_dict[mode]['s']['value']) + "')"
        #print(insert_query)
        #mycursor = mydb.cursor()
        #mycursor.execute(insert_query2)
        #mydb.commit()
        print(f'S arrival predicted at {tm+2}')
        print("Data Inserted in S table!")
    #time.sleep(2)

stream_picks.append(picks_dict)
#print("Picks Dictionary Output:::")
#print(stream_picks)


# for ch in channel:       
#     st_new = st_p.select(channel=ch)
#     for count, tm in enumerate(np.arange(info['stt'], info['endt']-window, overlap)):
#         st_slice = st_new.slice(tm, tm+window)
#         #print(st_slice)
#         batch_data = classifier.preproc(st_slice)
#         print(batch_data.shape)
#         #print(batch_data)
#         #print(batch_data.shape)
#         prediction = classifier.get_picks(batch_data)
#         picks_dict, _, pred_phases = classifier.post_proc(prediction=prediction, batch_data=batch_data, count=count, tm=tm, stt=info['stt'], f_act=info['f'], picks_dict=picks_dict)
#         print(pred_phases)
#         if pred_phases == 'p':
#             prediction_thread = Prediction_thread()
#             prediction_thread.start()
#             prediction_thread.join()
#             pred_s = prediction_thread.value
#             print(f'In Thread:: Pred_s: {pred_s}')
        
#         #time.sleep(2)

#     stream_picks.append(picks_dict)
#     print(stream_picks)

# start = 0
# window = 4
# for i in range(int((npts/40)/4)):
#     st_slice = st_p.slice(stt+start, stt+window)
#     #print(st_slice)
#     batch_data = classifier.preproc(st_slice)
#     #print(f'start: {stt+start}, End: {stt+window}')
#     #print(batch_data) 
#     #prediction = classifier.get_picks(batch_data)
#     #picks_dict, _ = classifier.post_proc(prediction=prediction, batch_data=batch_data, count=count, tm=tm, stt=info_p['stt'], f_act=info_p['f'], picks_dict=picks_dict)
#     start = window
#     window += 4
#     #print(i)

