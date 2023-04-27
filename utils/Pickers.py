"""
This class implements pickers defined by obspy and true arrivals in the required format by the Testbench class
"""
import numpy as np
from scipy.signal import find_peaks
import sys
import os
import obspy.signal.trigger as trigger
from SeedDataset import SeedDataSet
from Plotting import Plotting
import common as com
import time

class Pickers():
   def __init__(self, mode='truth', filt=0, proc_level = 'trace', eval_level = 'trace'):
     """
     mode: type of picker; 'truth': actual picks; 'pkbaer': obspy.signal.trigger.pk_baer picker; 'arpick': obspy.signal.trigger.ar_pick picker

     filt: if signal is to be filtered

     proc_level   : level for preprocessing; 'trace': performed on the whole trace; 'slice' or 'batch': performed on slices

     eval_level   : level for application; 'trace': performed on the whole trace; 'slice' or 'batch': performed on slices    
     """
     self.mode = mode
     self.filt = filt
     self.proc_level = proc_level
     self.eval_level = eval_level
     self.mode = mode
     self.single = 1
     self.get_time = 1
     self.win=20
     if self.mode == 'pkbaer':
       #self.proc_level = 'slice'
       #self.eval_level = 'slice'
       
       self.get_picks = self.apply_pkbaer
       self.post_proc= self.post_proc_pkbaer
     elif self.mode == 'arpick':
       #self.proc_level = 'slice'
       #self.eval_level = 'slice'
       self.get_picks = self.apply_arpick
       self.post_proc= self.post_proc_arpick
     elif self.mode == 'truth':
       self.get_picks = self.true_picks
       self.post_proc= self.post_proc_truth
     elif self.mode == 'stalta':
       self.get_picks = self.true_picks
       self.post_proc= self.post_proc_truth
       self.get_picks = self.sta_lta
       self.post_proc= self.post_proc_sta_lta
       self.on = 1.7
       self.off = 0.5
       self.trig = 0

   def preproc(self, stream):
     """
     This function applies preprocessing to the stream

     stream: obspy stream object
     """
     stream=stream.copy().detrend('demean')
     #stream=stream.detrend('demean')
     return stream 

   
   def true_picks(self, stream, ptime, stime):
     """
     This function prints P and S arrivals and returns the values

     stream: obspy stream object

     ptime: UTC datetime for P arrival

     stime: UTC datetime for S arrival

     returns utc datetime for p and s arrivals
     """
     print('Actual P arrival occur at {}'.format(ptime))
     print('Actual S arrival occur at {}'.format(stime))
     return ptime, stime

   def post_proc_truth(self, batch_data, prediction, count, tm, stt, f_act, picks_dict, triggers): 
     """
     This function places the true values in the picks dictionary

     prediction: picker output

     batch_data, count, tm, stt, f_act, picks_dict, triggers: added only for implementation uniformity for all pickers

     returns the appended picks and triggers dictionaries. 
     """ 
     ptime, stime = prediction
     picks_dict[self.mode]['p']['utc']= [ptime]
     picks_dict[self.mode]['p']['value']= [1.0]
     picks_dict[self.mode]['p']['pick'] = [com.get_loc(ptime, stt, f_act)] 
     picks_dict[self.mode]['s']['utc']= [stime]
     picks_dict[self.mode]['s']['value']= [1.0]
     picks_dict[self.mode]['s']['pick'] = [com.get_loc(stime, stt, f_act)] 
     return picks_dict, triggers
     
    
   def apply_pkbaer(self, stream, ptime=0, stime=0):
     """
     This function apply pkbaer phase pick from obspy.

     stream: obspy stream or trace object

     ptime, stime: added for uniformity

     returns p phase arrival from the current trace or stream
     """
     trace = stream if hasattr(stream, 'stats') else stream[-1] # if stream has traces with multiple channels take the last
     p_pick, _ = trigger.pk_baer(trace, trace.stats.sampling_rate, 
                            20, 60, 7.0, 12.0, 100, 100)  
     return p_pick

   def post_proc_pkbaer(self, batch_data, prediction, count, tm, stt, f_act, picks_dict, triggers):  
     """
     This function places the picks from pkbaer in the picks dictionary

     prediction: picker output

     batch_data, count, tm, stt, f_act, picks_dict, triggers: added only for implementation uniformity for all pickers

     returns the appended picks and triggers dictionaries. 
     """ 
     p_pick= prediction
     # check for valid result 
     if (p_pick>0) : 
       ptime = stt + (p_pick/f_act)
       ppick = p_pick  
       print('P.K. Briar method picked P at', ptime)
     else:
       ptime = 0.0
       ppick=0.0
       print('P.K. Briar method did not picked P')
     #print(picks_dict)
     picks_dict[self.mode]['p']['utc']= [ptime]
     picks_dict[self.mode]['p']['value']= [1.0]
     picks_dict[self.mode]['p']['pick'] = [ppick] 
     picks_dict[self.mode]['s']['utc']= [0.0]
     picks_dict[self.mode]['s']['value']= [0.0]
     picks_dict[self.mode]['s']['pick'] = [0.0] 
     return picks_dict, triggers

   # apply ar pick from obspy
   def apply_arpick(self, stream, ptime, stime):
     """
     This function apply ar_pick phase pick from obspy.

     stream: obspy stream or trace object

     ptime, stime: added for uniformity

     returns p and s phase arrivals from the current trace or stream
     """
     if hasattr(stream, 'stats'): # if single channel use only this value
       ar_ppick, ar_spick = trigger.ar_pick(stream.data, stream.data, 
                          stream.data, stream.stats.sampling_rate, 
                          1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
     else:
       ver = stream[-1] # the last channel will be vertical if available
       hor = []
       for tr in stream[:-1]: # the rest are horizonal 
         hor.append(tr)
       if len(hor) < 2:
         hor.append(tr)
       ar_ppick, ar_spick = trigger.ar_pick(hor[-1].data, hor[-2].data, 
                          ver.data, ver.stats.sampling_rate, 
                          1.0, 20.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
     return ar_ppick, ar_spick

   def post_proc_arpick(self, batch_data, prediction, count, tm, stt, f_act, picks_dict, triggers): 
     """
     This function places the picks from ar_pick in the picks dictionary

     prediction: picker output

     batch_data, count, tm, stt, f_act, picks_dict, triggers: added only for implementation uniformity for all pickers

     returns the appended picks and triggers dictionaries. 
     """ 
     ar_ppick, ar_spick = prediction[0], prediction[1]
     # check for valid result
     if (ar_ppick>0) : 
       ptime = stt + ar_ppick
       ppick= com.get_loc(ptime, stt, f_act)
       print('AR method picked P at', ptime)
     else:
       ppick = 0.0
       ptime = 0.0
       print('AR method did not picked P')

     # check for valid result
     if (ar_spick>0) : 
       stime = stt + ar_spick
       spick = com.get_loc(stime, stt, f_act)
       print('AR method picked S at', stime)
     else:
       spick = 0.0
       stime = 0.0
       print('AR method did not picked S')    
     #if self.get_time: print('arpic had an inference time of {} sec'.format(time.time() -t1))
     picks_dict[self.mode]['p']['utc']= [ptime]
     picks_dict[self.mode]['p']['value']= [1.0]
     picks_dict[self.mode]['p']['pick'] = [ppick] 
     picks_dict[self.mode]['s']['utc']= [stime]
     picks_dict[self.mode]['s']['value']= [1.0]
     picks_dict[self.mode]['s']['pick'] = [spick] 
     return picks_dict, triggers

   def sta_lta(self, stream, ptime=0, stime=0, sta_time=5, lta_time=0):
     """
     This function prints P and S arrivals and returns the values

     stream: obspy stream object

     ptime: UTC datetime for P arrival

     stime: UTC datetime for S arrival

     returns utc datetime for p and s arrivals
     """
     if lta_time==0: lta_time=self.win
     trace = stream if hasattr(stream, 'stats') else stream[-1]
     #sta = np.std(trace.data[-int(trace.stats.sampling_rate * sta_time):])
     #lta = np.std(trace.data[-int(trace.stats.sampling_rate * lta_time):])
     stalta_ratio = trigger.classic_sta_lta(trace.data, int(sta_time * trace.stats.sampling_rate), int(lta_time * trace.stats.sampling_rate))
     return stalta_ratio

   def post_proc_sta_lta(self, batch_data, prediction, count, tm, stt, f_act, picks_dict, triggers): 
     """
     This function places the true values in the picks dictionary

     prediction: picker output

     batch_data, count, tm, stt, f_act, picks_dict, triggers: added only for implementation uniformity for all pickers

     returns the appended picks and triggers dictionaries. 
     """ 
     for ind in range(len(prediction)):
       if self.trig==0 and prediction[ind] > self.on:
         self.trig=1
         ptime= stt+(ind/f_act)
         print(f'sta/lta picked P at {ptime}')       
         picks_dict[self.mode]['p']['utc'].append(ptime)
         picks_dict[self.mode]['p']['value'].append(1.0)
         picks_dict[self.mode]['p']['pick'].append(com.get_loc(ptime, stt, f_act))
       if prediction[ind] < self.off:        
         self.trig=0
     picks_dict[self.mode]['s']['utc']= [0.0]
     picks_dict[self.mode]['s']['value']= [0.0]
     picks_dict[self.mode]['s']['pick']= [0.0]
     return picks_dict, triggers
     

def main():
   """
   Small working example
   """
   dataPath = "..\\SeedData\\Jan2018-Jan2020_seed\\mseed4testing\\*.mseed"
   stations= ['AKA', 'NIL', 'SIMI']
   channels = ['BH1', 'BH2', 'BHZ']
   f = [40]
   dataset = SeedDataSet(dataPath)
   stream_list = dataset.stream_to_list(f=f, channels=channels, stations=stations)
   modes =['truth', 'pkbaer', 'arpick']   
   lines=[]
   for n, st in enumerate(stream_list[:1]):
     lines = com.myprint(lines, '*******************************************************************')
     lines = com.myprint(lines,'Stream : {} '.format(n))  
     info = com.get_info(st)
     ptime = info['stt'] + dataset.ptime
     stime = info['endt'] - dataset.stime
     for mode in modes:
       pick = Pickers(mode=mode)
       out = pick.get_picks(st, ptime=ptime, stime=stime)
       print(f'mode={mode} raw_output={out}') 
     
if __name__ == '__main__':
   main() 