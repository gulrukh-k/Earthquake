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
import os
import argparse

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
        self.runtime=0

    def run(self):
        t1 = time.time()
        pslice_proc = self.model.preproc(self.data)
        pred = self.model.predict(pslice_proc)
        self.pred = self.model.postproc(pred)
        self.runtime = (time.time()-t1)*1000
        print(f'In Thread:: {self.id}: {self.pred} ({(self.runtime):.2f})')   

def get_parser():
  parser = argparse.ArgumentParser(description='Phase classification and short-time prediction')
  
  parser.add_argument('--select', type=str, default='',
                    help='select seed data')

  parser.add_argument('--outdir', type=str, default='testing_results',
                    help='select seed data')

  parser.add_argument('--offset', type=float, default=5,
                    help='data slice interval')
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

  #Values to be set by user
  parser = get_parser()
  params = parser.parse_args()
  select = params.select
  offset = params.offset
  outdir = os.path.join('results', params.outdir)
  com.safe_mkdir(outdir)
  
  datapath = 'data\\new_test_data\\*.mseed'
  noisepath = 'data\\new_noise_data\\*.mseed'
  
  classifier = Classifier(mode=mode)
  classifier.model.summary()
  class_time =[]
  mpred = Predictor('m1')
  tpred = Predictor('t1') 
  m_time =[]
  t_time =[]
  loop_time =[]
  st = obspy.read(datapath)
  noise_st = obspy.read(noisepath)
  #mpred.test_pred(st, outdir=outdir, ptime=60, stime=200)
  #tpred.test_pred(st, outdir=outdir, ptime=60, stime=200)
  
  st = st.select(channel=channel)
  event_select = ['tp', 'tp_fn', 'fn', 'fn_fp', 'tp_err', 'early']
  noise_select = ['tn', 'fp']
  selection= {}
  selection['tp'] = [2, 6, 20, 27]
  selection['tp_fn'] =[8, 9, 11, 13, 17, 21, 26, 29]
  selection['fn'] = [0, 7, 10, 24, 28]
  selection['fn_fp'] = [1, 3, 5, 12, 14, 15, 16, 18, 22]
  selection['tp_err'] = [19, 23]  
  selection['early'] = [4, 25]

  selection['tn'] = [0, 1, 2, 3, 4, 5]
  selection['fp'] = [21, 22, 23, 24, 25]
 
  num=0
  if select=='':
    select = event_select+ noise_select
  else:
    select = [select]
  for sel in select:
    if sel in event_select:
        ptime=60
        stime =200
        sel_st = st
    elif sel in noise_select:
        ptime=0
        stime =0
        sel_st = noise_st
    seldir = os.path.join(outdir, sel)
    com.safe_mkdir(seldir)   
    new_stream=obspy.core.stream.Stream() 
    for sl in selection[sel]:      
      new_stream.append(sel_st[sl]) 
    for tr in new_stream: 
        #num = selection[select]
        #st_p = st[num]
    
        st_p = tr
        info = com.get_info(st_p)
        stt, endt = info['stt'], info['endt']
        act_f = info['f']
        if (ptime > 0): 
          print(f'P arrival at {stt+60}',)
        if (stime > 0): 
          print(f'S arrival at {endt-200}')
        pphase=0
        p_npts =0
        sphase =0
        prev_peak = 0
        act_p_npts = info['f'] * ptime
        act_s_npts = info['f'] * (endt-stt-stime)
        ptriggers=[]
        striggers=[]
        speaks=[]
        for count, tm in enumerate(np.arange(info['stt'], info['endt']-window, offset)):
          loop_init=time.time()
          st_slice = st_p.slice(tm, tm+window)
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
            print(f'*************************************************************************')
            print(f'P arrival predicted at {tm}')
            mprediction_thread = Prediction_thread(id='Amplitude', data=st_slice, model=mpred)
            tprediction_thread = Prediction_thread(id='time',data=st_slice, model=tpred)
            mprediction_thread.start()        
            tprediction_thread.start()
            mprediction_thread.join()
            tprediction_thread.join()
            m_time.append(mprediction_thread.runtime)
            t_time.append(tprediction_thread.runtime)
            print(f'S wave with amplitude {mprediction_thread.pred:.02f} is expected after {tprediction_thread.pred/act_f:.02f} seconds from P arrival')
            pred_s = mprediction_thread.pred
            pred_t = tprediction_thread.pred
            if pphase==0:
              pphase = 1 
              p_arrival = tm
              p_npts = com.npts(p_arrival-info['stt'], f = act_f)
              ptriggers.append(p_npts)
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
              s_arrival = tm
              s_npts = com.npts(s_arrival-info['stt'], f = act_f)
              striggers.append(s_npts)
              if plot: com.plot_trigger(st_p, [s_npts, s_npts + 10000])
          
          if sphase >0 and (tm - s_arrival) < 100:
           peak_amp = np.max(np.absolute(st_slice.data))
           if peak_amp > prev_peak: 
             act_peak = peak_amp      
             peak_npts = np.argmax(peak_amp)
             speaks.append(peak_npts + int((tm-info['stt'])*info['f']))         
             if p_npts: 
               p_peak_npts = s_npts - p_npts + peak_npts
             else:
               p_peak_npts = s_npts - act_p_npts + peak_npts
             prev_peak = peak_amp
         
          if sphase >0:
            if (tm - s_arrival) > 100:
              sphase, pphase =0, 0
              print(f'actual S amplitude {act_peak} arrived {p_peak_npts/act_f} seconds after the detcted P')
          loop_time.append((time.time()-loop_init)*1000)
        com.plot_trigger_ps(st_p, ptrigger=ptriggers, true_ptrigger=[act_p_npts],
                      strigger=striggers, true_strigger=[act_s_npts],
                      outdir = seldir, name = str(num))    
        num+=1
  class_time = np.mean(np.array(class_time))
  m_time = np.mean(np.array(m_time))
  t_time = np.mean(np.array(t_time))
  loop_time = np.mean(np.array(loop_time))
  print(f'The classifier run time = {class_time} ms')
  print(f'The M predictor run time = {m_time} ms')
  print(f'The t predictor run time = {t_time} ms')
  print(f'The loop run time = {loop_time} ms')
if __name__ == '__main__':
   main() 