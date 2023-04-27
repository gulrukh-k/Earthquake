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
from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import requests
#from websocket import create_connection
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
    def __init__(self, model, data=0, id=0):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.id = id
        self.data=data
        self.model=model
        self.pred=0
        self.runtime=0

    def run(self):
        t1 = time.time()
        pslice_proc = self.model.preproc(self.data)
        pred = self.model.predict(pslice_proc)
        self.pred = self.model.postproc(pred)
        self.runtime = (time.time()-t1)*1000
        print(f'In Thread:: {self.id}: {self.pred} ({(self.runtime):.2f})')  

class API_Thread(Thread):

    def __init__(self, data, id):
        Thread.__init__(self)
        self.data = data
        self.id = id

    def run(self):
        call_api(self.data, self.id)
    
def call_api(data, id):
    if id == 0:
      req1 = 'http://localhost/github/sarmad_dev/EarthQuakePHP/APIs/api_pwave_to_db.php?pwave=true&unique_id={}&station_id=S1&p_arrival_time={}&expected_S_time={}&predicted_amplitude={}'.format(data['unique_id'],data["p_arrival_time"],data["expected_S_time"],data["predicted_amplitude"])  
      try:
        r = requests.post(req1)
        print('data sent to API successfully!')
      except:
        print("can't get to the api_1")
    if id == 1:
      print(data)
      req2 = 'http://localhost/github/sarmad_dev/EarthQuakePHP/APIs/api_pwave_to_db.php?swave=true&unique_id={}&station_id={}&S_arrival_time={}&Detected_S_amplitude={}'.format(data['unique_id'], data['station_id'], data['S_arrival_time'], data['Detected_S_amplitude'])
      try:
        r = requests.post(req2)
        print('data sent to API successfully!')
      except:
        print("can't get to the api_2")
    if id == 2:
        print(data)
        print('***************************************************')
        try:
            ws = create_connection("ws://localhost:8080")      
            json_object = json.dumps(data)
            ws.send(json_object)
            print("sent")
            ws.close
        except:
            print("can't connect to the socket")

def print_timing(t_list, labels):
  for t, label in zip(t_list, labels):
    if type(t)==list: t = np.array(t)
    t=np.mean(t)
    print(f'The {label} run time = {t} ms')

def start(model):
  input_shape = model.get_config()["layers"][0]["config"]["batch_input_shape"]
  model_in = np.expand_dims(np.zeros(input_shape[1:]), 0)
  _=model.predict(model_in)

def get_category(id, title=''):
  if 't' in id: title=title +'True '
  if 'f' in id: title=title +'False '
  if 'p' in id: title=title +'Positive '  
  if 'n' in id: title=title +'negative '
  return title

def get_parser():
  parser = argparse.ArgumentParser(description='Phase classification and short-time prediction')
  
  parser.add_argument('--select', type=str, default=['tp', 'tn'],
                    help='select seed data')

  parser.add_argument('--outdir', type=str, default='results_animation_all',
                    help='select seed data')

  parser.add_argument('--offset', type=float, default=4,
                    help='data slice interval')

  parser.add_argument('--interval', type=float, default=0.00001,
                    help='data update interval')

  parser.add_argument('--user', action='store_true')

  return parser        

def main():
  mode = 'stftcnn7'
  window = 4 # sec  
  channel = 'HHZ'
  thresh ={}
  thresh['p'] = 0.9
  thresh['s'] = 0.9
  thresh['n'] = 0.0
  plot=0
  plt.ion()
  #Values to be set by user
  parser = get_parser()
  params = parser.parse_args()
  select = params.select
  offset = params.offset
  interval = params.interval
  user = params.user
  outdir = os.path.join('results', params.outdir)
  com.safe_mkdir(outdir)
  
  datapath = 'data\\new_test_data\\*.mseed'
  #noisepath = 'data\\new_noise_data\\*.mseed'
  noisepath = 'data\\pesh_noise_marker2\\*.mseed'
  classifier = Classifier(mode=mode)
  classifier.model.summary()
  start(classifier.model)
  
  class_time =[]
  mpred = Predictor('m1')
  start(mpred.model)
  tpred = Predictor('t1')
  start(tpred.model) 
  m_time =[]
  t_time =[]
  loop_time =[]
  st = obspy.read(datapath)
  noise_st = obspy.read(noisepath)
  
  st = st.select(channel=channel)
  noise_st = noise_st.select(channel=channel)
  
  event_select = ['tp', 'tp_fn', 'fn', 'fn_fp', 'tp_err', 'early']
  noise_select = ['tn', 'fp']
  selection= {}
  selection['tp'] = [6]#2, 6, 20, 27]
  selection['tp_fn'] =[8]#, 9, 11, 13, 17, 21, 26, 29]
  selection['fn'] = [0]#, 7, 10, 24, 28]
  selection['fn_fp'] = [1]#, 3, 5, 12, 14, 15, 16, 18, 22]
  selection['tp_err'] = [19]#, 23]  
  selection['early'] = [4]#, 25]

  #selection['tn'] = [0, 1, 2, 3, 4, 5]
  #selection['fp'] = [21, 22, 23, 24, 25]
  selection['tn'] = [2]#, 3, 8, 9, 10, 15]
  selection['fp'] = [0]#, 1, 4, 5, 6]
  
  while True:
    if user:
      user_in = input('Select input (tp, tn, fp, fn):')
      if user_in == "":
        if len(class_time)>0:
          print_timing([class_time, m_time, t_time, loop_time], ['classifier', 'M predictor', 't predictor', 'loop']) 
        exit()
      else:
        if (user_in in event_select) or (user_in in noise_select):
          select = [user_in]
          print(f'You selected {get_category(user_in)}')
        else:
          print(f'Unknown category {user_in}')
          exit()  
              
    if select=='':
      select = event_select+ noise_select
    elif type(select)is not list:
      select = [select]
    for sel in select:
      if sel in event_select:
        ppos=60
        spos =200
        sel_st = st
      elif sel in noise_select:
        ppos=0
        spos =0
        sel_st = noise_st
      elif sel == 'noise_all':
        ppos=0
        spos =0
        sel_st = noise_st
      else:
        print(f'Unknown category {sel}') 
        continue  
      seldir = os.path.join(outdir, sel)
      com.safe_mkdir(seldir)   
      new_stream=obspy.core.stream.Stream() 
      if sel in selection:
        for sl in selection[sel]:      
          new_stream.append(sel_st[sl])
      else:
        new_stream = sel_st 
      num=0
      if not user: input(f"Press Enter to continue with {get_category(sel)}...")
      #fig1, axes = plt.subplots(2, 1, figsize=(8, 5))
      for tr in new_stream[::]: 
        st_p = tr
        info = com.get_info(st_p)
        stt, endt = info['stt'], info['endt']
        act_f = info['f']
        if (ppos > 0): 
          ptime = stt+ppos
          print(f'P arrival at {ptime}',)
        else: ptime=0
        if (spos > 0): 
          stime = endt-spos
          print(f'S arrival at {stime}')
        else: stime=0
        pphase=0
        p_npts =0
        sphase =0
        prev_peak = 0
        l1=0
        act_p_npts = info['f'] * ppos
        act_s_npts = info['f'] * (endt-stt-spos)
        ptriggers=[]
        striggers=[]
        p_arrivals=[]
        s_arrivals= []
        speaks={}
        speaks['m']=[]
        speaks['t']=[]
        m_pred=[]
        t_pred =[]
        
        for count, tm in enumerate(np.arange(info['stt'], info['endt']-window, offset)):
          loop_init=time.time()
          st_slice = st_p.slice(tm, tm+window)
          
          mprediction_thread = Prediction_thread(id='Amplitude', data=st_slice, model=mpred)
          tprediction_thread = Prediction_thread(id='time', data=st_slice, model=tpred)
          slice_npts = int((tm - info['stt'])* info['f'] )
          t1= time.time()
          batch_data = classifier.preproc(st_slice)              
          prediction = classifier.get_picks(batch_data)
          
          class_time.append((time.time() - t1)*1000)
          pred_val = np.max(prediction)
          pred_phase = classifier.phases[np.argmax(prediction)]
          if pred_val > thresh[pred_phase]:
            pred_phase = pred_phase
          else:
            pred_phase = 'n'
          if pred_phase == 'p':
            p_info = {}
            print(f'*************************************************************************')
            print(f'P arrival predicted at {tm}')
            dt = datetime.now()
            unique_id = int(f"{dt.year}{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}{dt.second:02d}{dt.microsecond:06d}")
            p_info['unique_id'] = unique_id
            time_str = str(tm)
            p_info['p_arrival_time'] = time_str[0:10]+'%'+time_str[11:19]
            p_info['station_id'] = 'S1' 
            mprediction_thread.start()        
            tprediction_thread.start()
            mprediction_thread.join()
            tprediction_thread.join()
            if (mprediction_thread.pred > 0): m_pred.append(mprediction_thread.pred)
            if (tprediction_thread.pred > 0): t_pred.append(tprediction_thread.pred/act_f)
            m_time.append(mprediction_thread.runtime)
            t_time.append(tprediction_thread.runtime)
            print(f'S wave with amplitude {mprediction_thread.pred:.02f} is expected after {tprediction_thread.pred/act_f:.02f} seconds from P arrival')
            pred_s = mprediction_thread.pred
            pred_t = tprediction_thread.pred
            p_info['predicted_amplitude'] = "{:.2f}".format(pred_s)
            p_info['expected_S_time'] = "{:.2f}".format(pred_t)
            api_call_p = API_Thread(p_info, id=0)
            api_call_p.start()
            print('main thread is still continue')
            web_socket_ = API_Thread(p_info, id=2)
            web_socket_.start()
            if pphase==0:
              pphase = 1 
              p_arrival = tm+(window/2)
              p_npts = com.npts(p_arrival-info['stt'], f = act_f)
              ptriggers.append(p_npts)
              p_arrivals.append(p_arrival)
              if plot: com.plot_trigger(st_p, p_npts)
              #insert_query = "insert into p_pred (Pwave,PredS) values ('" + str(picks_dict[mode]['p']['value']) + "','" + str(pred_s) + "')"
              #print(insert_query)
              #mycursor = mydb.cursor()
              #mycursor.execute(insert_query)
              #mydb.commit()
              #print("Data Inserted in P table!")
        
          if pred_phase == 's':
            print(f'*************************************************************************')
            print(f'S arrival predicted at {tm}')
            s_info = {}
            s_info['unique_id'] = p_info['unique_id']
            time_str = str(tm)
            s_info['S_arrival_time'] = time_str[0:10]+'%'+time_str[11:19]
            s_info['station_id'] = 'S1'
            #insert_query2 = "insert into detecteds (DetectedS) values ('" + str(picks_dict[mode]['s']['value']) + "')"
            #print(insert_query)
            #mycursor = mydb.cursor()
            #mycursor.execute(insert_query2)
            #mydb.commit()
            #print("Data Inserted in S table!")
            #sslice = st_p.slice(s_arr, stime+100)
            #max_amp = np.max(np.absolute(sslice.data))
            if sphase==0: 
              sphase = 1
              s_arrival = tm+(window/2)
              s_npts = com.npts(s_arrival-info['stt'], f = act_f)
              striggers.append(s_npts)
              s_arrivals.append(s_arrival)
              if plot: com.plot_trigger(st_p, [s_npts, s_npts + 10000])
          if sphase >0 and (tm - s_arrival) < 100:
            peak_amp = np.max(np.absolute(st_slice.data))
            if peak_amp > prev_peak: 
              act_peak = peak_amp      
              peak_npts = np.argmax(peak_amp)
              #speaks.append(peak_npts + int((tm-info['stt'])*info['f']))         
              if p_npts: 
                p_peak_npts = s_npts - p_npts + peak_npts
              else:
                p_peak_npts = s_npts - act_p_npts + peak_npts
              prev_peak = peak_amp
         
          if (sphase >0):
             if (tm - s_arrival) > 100:
               sphase, pphase =0, 0
               speaks['m'].append(act_peak)
               speaks['t'].append(p_peak_npts/act_f)
               print(f'actual S amplitude {act_peak} arrived {p_peak_npts/act_f} seconds after the detcted P')
               s_info['Detected_S_amplitude'] = act_peak
               api_call = API_Thread(s_info, id=1)
               api_call.start()
          loop_time.append((time.time()-loop_init)*1000)
          slice_sum = st_slice if count==0 else st_p.slice(stt, tm+window) 
          if count==0: fig1, axes = plt.subplots(2, 1, figsize=(8, 5)) 
          title='Peshawar Station:   '
          title =get_category(sel, title)
          if ptime: 
            title = title + f'   Earthquake P wave arriving at {ptime.isoformat()}'
          else:
            title = title + 'Noise data'
          fig1.canvas.set_window_title(title)      
          l1 = com.plot_trigger_animate(st_slice, slice_sum, fig1, axes, count, l1, ptrigger=p_arrivals, true_ptrigger=[ptime],
                 m_pred=m_pred, t_pred=t_pred, strigger=s_arrivals, true_strigger=[stime], speaks=speaks)
          
          fig1.canvas.draw()
          plt.tight_layout()
          #fig1.canvas.flush_events()
          plt.pause(interval)
        fig1.savefig(os.path.join(seldir, f'event{str(num)}.png'))       
        plt.close()
        num+=1
    if not user:
      print_timing([class_time, m_time, t_time, loop_time], ['classifier', 'M predictor', 't predictor', 'loop']) 
      exit()    
  
if __name__ == '__main__':
   main() 