"""
This class uses the SeedDataset, Pickers, and Classifier classes to compare the performance on continous trace

A sliding window over the trace is created corressponding to respective algorithm and picks are obtained

These picks are validated and compared
"""

import numpy as np
import os
from SeedDataset import SeedDataSet
from ARRU import AppARRU
from Pickers_slice import Pickers as slice_Pickers
from Pickers import Pickers
from Classifier import Classifier
from CapsPhase import CapsPhase
import common as com
import time
import h5py
import sklearn.metrics as metrics

class Testbench():
   def __init__(self, start = 0, end= 0, interval =10, pick_modes=[], phases = ['p', 's', 'n'], get_time=1,  tolerance=0.5,
                if_full=0,  dataset=None):
     """
     start: seconds from trace trace when algorithm is to be run

     end: seconds from trace trace when algorithm is to be run

     interval: interval between algorith application

     pick_modes: a list of names representing agorithms to apply. Following modes are available:
         
     - 'truth' : true arrivals if available. If not available then evaluation will not be done but the picks can be plotted.
  
     - 'pkbaer' : picker for P arrival available from https://docs.obspy.org/packages/autogen/obspy.signal.trigger.pk_baer.html 

     - 'arpick' : picker for P and S arrival available from https://docs.obspy.org/packages/autogen/obspy.signal.trigger.ar_pick.html

     - 'arrupick': picker for P and S arrival available from https://github.com/tso1257771/Attention-Recurrent-Residual-U-Net-for-earthquake-detection/tree/main/pretrained_model/paper_model_ARRU_20s

     - 'arrudetect': picker for P and S arrival available from https://github.com/tso1257771/Attention-Recurrent-Residual-U-Net-for-earthquake-detection/tree/main/pretrained_model/multitask_ARRU_20s

     - 'stftcnn1' - 'stftcnn6' : different versions of cnn models trained on stft of seismic traces

     phases: list of selected phases or classes to evaluate. Currently including 'p', 's', 'n' and 'tail' can be selected.

     dataset: SeedDataset object

     self.time: if timing the algorithms

     self.full_trace: 0: picks only evaluated near actual only for ARRU; 1: whole trace is evaluated

     self.maxwin: maximum possible size for window

     self.iter_no: number of current iterations
     self.pos_pick_lim": maximum error tolerance in sec for positive pick
 
     self.models_dict: dict for all available approaches as objects

     self.picks_dict: dict to hold the output of the approaches

     self.eval_dict: dict for validation metrics

     self.time_dict: dict for timing data

     self.triggers: dict for trigger related data

     self.iterations: dict to hold iterations for each algorithm
     """
     self.start = start # offset from the starting point
     self.end = end     # offset from the end point
     self.interval = interval   # distance in sec between consecutive data slices for algorithm application  
     self.modes = pick_modes  # the picking approaches to test
     self.time = get_time # if timing the algorithms
     self.full_trace = if_full # 0: picks only evaluated near actual only for ARRU; 1: whole trace is evaluated
     self.maxwin = 20 # maximum possible size for slice
     self.iter_no = 0  # iterations of test
     self.tolerance = tolerance
     self.pos_pick_lim = self.interval + tolerance  # maximum error tolerance in sec  
     self.phases = phases # phases included in validation
     self.models_dict={} # dict of all available approaches as objects
     self.picks_dict={}  # dict to hold the output of the approaches
     self.eval_dict={}   # dict for validation metrics
     self.time_dict = {}
     self.triggers = {}
     self.iterations = {} # iterations for slides
     self.triggers['sta/lta'] = {}
     # initialize the dictionaries 
     for mode in self.modes:
       if 'arru' in mode:
         #if mode=='arrupick':
         #  pthresh, sthresh=0.5, 0.4
         #elif mode =='arrudetect':
         #  pthresh, sthresh=0.5, 0.1
         #arru_model = AppARRU(mode, single=0, pthresh=pthresh, sthresh = sthresh)
         arru_model = AppARRU(mode)
         self.models_dict[mode]={}
         self.models_dict[mode]['model']=arru_model
       if mode in ['truth', 'pkbaer', 'arpick', 'stalta']:         
         truth_model = Pickers(mode=mode)
         self.models_dict[mode]={}
         self.models_dict[mode]['model']=truth_model 
       if mode in ['pkbaer_slice', 'arpick_slice', 'stalta_slice']:
         truth_model = slice_Pickers(mode=mode)
         self.models_dict[mode]={}
         self.models_dict[mode]['model']=truth_model 
       if 'stftcnn' in mode:
         #pthresh, sthresh=0.5, 0.7
         stft_model = Classifier(mode)#, pthresh=pthresh, sthresh = sthresh)
         self.models_dict[mode]={}
         self.models_dict[mode]['model']=stft_model  
       if 'caps' in mode:
         pthresh, sthresh=0.5, 0.5
         caps_model = CapsPhase(mode, pthresh=pthresh, sthresh = sthresh)
         
         self.models_dict[mode]={}
         self.models_dict[mode]['model']=caps_model                   
       self.picks_dict[mode]={}
       self.eval_dict[mode]={}
       self.time_dict[mode]={}
       self.iterations[mode]=[]
       self.triggers['sta/lta'][mode]=[]
     self.init_validation()
     self.init_time()

   def init_picks(self):
      """
      This function initializes dictionary structure for algorithm outputs 
      """
      for mode in self.modes:
        for phase in self.phases:
          self.picks_dict[mode][phase]={}
          self.picks_dict[mode][phase]['pick']=[]  # sample no. of the pick
          self.picks_dict[mode][phase]['value']=[] # peak value
          self.picks_dict[mode][phase]['utc']=[]   # time for the pick
          self.picks_dict[mode][phase]['out']=[]   # all values
          self.picks_dict[mode][phase]['out_t']=[]   # time for all values
          self.picks_dict[mode][phase]['out_loc']=[]   # npts for all values

   def init_validation(self):
      """
      This function initializes dictionary structure for validation results 
      """
      for mode in self.modes:
        for phase in self.phases:
          self.eval_dict[mode][phase]={}
          self.eval_dict[mode][phase]['mae']=[]  # MAE
          self.eval_dict[mode][phase]['accuracy']=[] # MAPE
          self.eval_dict[mode][phase]['precision']=[]   # list for precision
          self.eval_dict[mode][phase]['recall']=[]   # list for recall
          self.eval_dict[mode][phase]['f1']=[]   # list for f1
          self.eval_dict[mode][phase]['picks']=0   # counter for the pick rate

   
   def init_time(self):
      """
      This function initializes dictionary structure for timing results 
      """
      for mode in self.modes:
        self.time_dict[mode]['pre']=[]  # preprocessing time
        self.time_dict[mode]['eval']=[] #  application time
        self.time_dict[mode]['post']=[] # post processing time
        self.time_dict[mode]['count']=[] # iterations
       
   def get_picks(self, stream, ptime=0, stime=0, log=None):
     """
     This function obtains picks in the internal picks dictionary for a stream based on all selected modes. Evaluation and timing results are also saved in respective dicts with corressponding text log generated.

     stream: obspy stream object

     ptime: time in seconds from trace starting point

     stime: time in seconds from trace end point 

     log: text list to which further info can be appended

     returns updated log
     """
     first_sample=1
     hist_avg = 0
     self.init_picks() # initialize for each stream
     info = com.get_info(stream) # info of original stream 
       
     for mode in self.models_dict:
       app = self.models_dict[mode]['model'] # the algorithm object
       self.iterations[mode].append(0)
       if app.proc_level == 'trace':
          if self.time: t_init = time.time()
          st_proc = app.preproc(stream) # if preprocessing is applied for the whole trace
          if self.time:  self.time_dict[mode]['pre'].append(time.time()-t_init)
          print('stream processed') 
       
       # if model is to be evaluated for the whole trace then get picks
       if self.models_dict[mode]['model'].eval_level == 'trace':
         if self.time: t_init = time.time()
         model_out =  self.models_dict[mode]['model'].get_picks(stream=st_proc, ptime=ptime, stime=stime)
         self.picks_dict, self.triggers = app.post_proc(batch_data=stream, prediction=model_out, count=0, tm=info['stt']+self.start, 
                                         stt=info['stt'], f_act=info['f'], picks_dict=self.picks_dict, triggers=self.triggers)
         self.iterations[mode][self.iter_no]+=1
         if self.time:  
            self.time_dict[mode]['eval'].append(time.time()-t_init)
            self.time_dict[mode]['post'].append(0)  
         self.time_dict[mode]['count'].append(1) 
     # check if there are sufficient samples available for iterations 
     if (info['stt']+self.start+self.maxwin< info['endt']-self.end):
       #iter = np.arange(info['stt']+self.start, info['endt']-self.maxwin - self.end, self.interval)
       iter = np.arange(info['stt']+self.start+self.maxwin, info['endt'] - self.end, self.interval)
     else:
       iter = [info['stt']+self.start+self.maxwin]
     
     for count, tm in enumerate(iter): 
       
       for mode in self.models_dict:
         app = self.models_dict[mode]['model'] # the algorithm object
         if app.eval_level == 'trace': # if algorithm has been evaluated for the full trace then bypass the rest
           continue       
         #slice the trace
         if app.proc_level == 'trace':           
           st_slice = st_proc.slice(tm-app.win, tm)
           batch_data =app.get_batch(st_slice)           
         else:
           if self.time: t_init = time.time()         
           st_slice = stream.slice(tm-app.win, tm)
           
           batch_data = app.preproc(st_slice)            # if preprocessing is performed on each slice 
           if self.time: 
             if count == 0 :  self.time_dict[mode]['pre'].append(0)
             self.time_dict[mode]['pre'][self.iter_no]+=time.time()-t_init      
         
         if self.time: t_init = time.time()   
         # get picks
                  
         model_out = app.get_picks(batch_data)
         if self.time: 
             if count == 0 :  
                self.time_dict[mode]['eval'].append(0)
                self.time_dict[mode]['count'].append(0)
             self.time_dict[mode]['eval'][self.iter_no]+=time.time()-t_init   
             self.time_dict[mode]['count'][self.iter_no]+=1
         if self.time: t_init = time.time()
         self.picks_dict, self.triggers = app.post_proc(batch_data=batch_data, prediction=model_out, count=count, tm=tm, 
                                         stt=info['stt'], f_act=info['f'], picks_dict=self.picks_dict, triggers=self.triggers)
         self.iterations[mode][self.iter_no]+=1
         if self.time: 
             if count == 0 :  
                self.time_dict[mode]['post'].append(0)
             self.time_dict[mode]['post'][self.iter_no]+=time.time()-t_init   
             
         #self.iterations[mode][self.iter_no]+=1
     for mode in self.picks_dict:
       for p_cnt, phase in enumerate(self.phases):
         if len(self.picks_dict[mode][phase]['pick'])>0: # if picks are available
           # if a single pick is desired then choose the one with maximum value
           if self.models_dict[mode]['model'].single: #                               
             arg = np.argmax(self.picks_dict[mode][phase]['value'])
             self.picks_dict[mode][phase]['pick']=[self.picks_dict[mode][phase]['pick'][arg]]
             self.picks_dict[mode][phase]['utc']=[self.picks_dict[mode][phase]['utc'][arg]]
             self.picks_dict[mode][phase]['value']=[self.picks_dict[mode][phase]['value'][arg]]
         # if the pick is empty then replace with zero
         else:
           self.picks_dict[mode][phase]['pick']=[0.0]
           self.picks_dict[mode][phase]['utc']=[0.0]
           self.picks_dict[mode][phase]['value']=[0.0]
        
     if log: log = self.evaluate_st(log)
     self.iter_no+=1
     
     return log

   # display the picks on screen
   def display_picks(self):
     label = ['P', 'S', 'noise', 'tail']
     for mode in self.picks_dict:
       for i, pick in enumerate(self.picks_dict[mode]):
         if len(self.picks_dict[mode][pick]['utc'])>0 and self.picks_dict[mode][pick]['utc'][0] > 0:
           if len(self.picks_dict[mode][pick]['utc'])==1:
             print('The {} predicted {} arrival at {}'.format(mode, label[i], self.picks_dict[mode][pick]['utc'][0]))
           else:
             print('The {} predicted {} arrival at:'.format(mode, label[i]))
             print(*self.picks_dict[mode][pick]['utc'], sep=',  ')
 
   def get_phase_array(self, mode, phase, picks, flag=0):
     """
     This function obtains a boolean array corresponding to a certain mode and phase for all test iterations. 

     The iteration where the input slice contains time in the picks list, is represented by one while the rest are zero.

     mode: mode for which the array is to be obtained

     phase: current phase

     picks: list with utcdatetime objects 

     returns 
     """
     win =  self.models_dict[mode]['model'].win # model window 
     limits=[-win/2, win/2] # the data limits for a model recognizing only phase arrival
     # check if class has predefined limits and the size of the phase needs to be extended: used to make the true phase when only the arrival is provided
     if hasattr(self.models_dict[mode]['model'], 'phase_lims') and flag==1: 
       limits = self.models_dict[mode]['model'].phase_lims[phase]  # then take these limits                   
     phase_array=[]           
     for i, t in enumerate(self.picks_dict[mode][phase]['out_t']): # get all time objects denoting centers of input slice
       t1 = t-(win/2) # start of first slice
       t2 = t+(win/2) # end of first slice
       picked = 0
       if picks != [0.0]:
         for pick in picks:
           if ((pick - abs(limits[0])-self.interval)<=t1) and ((pick + abs(limits[1])+self.interval)>=t2): # check to see if the predicted pick data is in the current slice
             picked=1
         #print(t1, t2, pick - abs(limits[0]), pick +abs(limits[1]), picked)   
       phase_array.append(picked)     
     return np.array(phase_array)          
          

   def evaluate_st(self, lines):
     """
     This function performs evaluation for a stream written as a table to screen and appended to list of text.
     
     lines: list of text

     returns appended text
     """
     lines = com.myprint(lines, '{0:12}{1:6}{2:>12}{3:>12}{4:>12}{5:>12}{6:>12}{7:>12}'.format('model', 'phase', 'mae (sec)', 'accuracy', 'precision', 'recall', 'F1', 'picks'))

     lines = com.myprint(lines, '---------------------------------------------------------------------------------------')

     if 'truth' in self.picks_dict:
       for mode in self.picks_dict:         
         for phase in self.picks_dict[mode]:
           if phase != 'n':          
             mae_list=[]
             picks=0
             #print(self.picks_dict[mode][phase]['utc'])
             if len(self.picks_dict[mode][phase]['utc'])>0 and self.picks_dict[mode][phase]['utc'][0] > 0:
               if hasattr(self.models_dict[mode]['model'], 'phase_lims'): # for class with a predefined limits
                 limits = self.models_dict[mode]['model'].phase_lims[phase] 
               else:
                 limits = [0, 0]
               for t in self.picks_dict[mode][phase]['utc']:
                 mae = self.picks_dict['truth'][phase]['utc'][0]-t 
                 if (mae>=limits[0]-self.tolerance) and (mae < limits[1] + self.tolerance): picks+=1
                 mae_list.append(np.absolute(mae))
               mae=min(mae_list)
               
             else:
               mae = -999
             if picks >= 1: self.eval_dict[mode][phase]['picks']+= 1
             #print(mae, self.eval_dict[mode][phase]['picks'])
             if self.models_dict[mode]['model'].eval_level=='trace': 
            
               acc, f1, precision, recall = 0, 0, 0, 0               
               #continue
             else:
               true_phase = self.get_phase_array(mode, phase, self.picks_dict['truth'][phase]['utc'], flag=1)
               pred_phase = self.get_phase_array(mode, phase, self.picks_dict[mode][phase]['utc'])
               #print('true', true_phase) 
               #print('pred', pred_phase) 
               
               acc = metrics.accuracy_score(true_phase, pred_phase)
               f1 = metrics.f1_score(true_phase, pred_phase)
               precision = metrics.precision_score(true_phase, pred_phase)
               recall = metrics.recall_score(true_phase, pred_phase)
               #print(acc, f1, precision, recall)
             self.eval_dict[mode][phase]['precision'].append(precision)
             self.eval_dict[mode][phase]['recall'].append(recall)
             self.eval_dict[mode][phase]['f1'].append(f1)
             self.eval_dict[mode][phase]['mae'].append(mae)
             self.eval_dict[mode][phase]['accuracy'].append(acc)
        
       for phase in self.phases:
         if phase == 'p' or phase == 's':
           for mode in self.picks_dict:
             
             lines = com.myprint(lines, '{0:<12}{1:<6}{2:>12.2f}{3:>12.2f}{4:>12.2f}{5:>12.2f}{6:>12.2f}{7:>12.2f}'.format(mode, phase, 
                     self.eval_dict[mode][phase]['mae'][-1], self.eval_dict[mode][phase]['accuracy'][-1],
                     self.eval_dict[mode][phase]['precision'][-1], self.eval_dict[mode][phase]['recall'][-1], self.eval_dict[mode][phase]['f1'][-1], self.eval_dict[mode][phase]['picks'])) 
           lines = com.myprint(lines, '---------------------------------------------------------------------------------------')
       if self.time:
         lines = com.myprint(lines, '---------------------------------------------------------------------------------------')
         lines = com.myprint(lines, f'--------------------------------------TIMING {self.iter_no} ---------------------------------------') 
         lines = com.myprint(lines, '{0:10}{1:>20}{2:>20}{3:>20}{4:>10}'.format('model', 'preproc (ms)', 'evaluation (ms)', 'postproc (ms)', 'msec/trace'))
         lines = com.myprint(lines, '{0:10}{1:>10}{2:>10}{3:>10}{4:>10}{5:>10}{6:>10}'.format(' ', 'total', 'avg', 'total', 'avg', 'total', 'avg'))
         
         for mode in self.picks_dict:
           avg_pre = 1000 *(self.time_dict[mode]['pre'][self.iter_no]/ self.time_dict[mode]['count'][self.iter_no])
           avg_eval = 1000 *(self.time_dict[mode]['eval'][self.iter_no]/self.time_dict[mode]['count'][self.iter_no])
           avg_post = 1000 *(self.time_dict[mode]['post'][self.iter_no]/self.time_dict[mode]['count'][self.iter_no])
           lines = com.myprint(lines, '{:10}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}'.format(mode, 
                     self.time_dict[mode]['pre'][self.iter_no]*1000, avg_pre, 
                     self.time_dict[mode]['eval'][self.iter_no]*1000, avg_eval,
                     self.time_dict[mode]['post'][self.iter_no]*1000, avg_post,
                     avg_pre + avg_post + avg_eval))
         lines = com.myprint(lines, '---------------------------------------------------------------------------------------')   
             
     else:
       lines = com.myprint(lines, ' Validation cannot be formed as true arrivals are not provided')
       lines = com.myprint(lines, '---------------------------------------------------------------------------------------')
     return lines  

   def evaluate(self, lines):
     """
     This function performs evaluation for the entire data. Writes as a table to screen and appendeds log to list of text.
     
     lines: list of text

     returns appended text
     """
     if 'truth' in self.picks_dict: 
       lines = com.myprint(lines, '------------------------------------FINAL VALIDATION-----------------------------------')
       if self.time:
         lines = com.myprint(lines, '{0:12}{1:6}{2:>12}{3:>12}{4:>12}{5:>10}{6:>10}{7:>10}{8:>12}'.format('model', 'phase', 'mae (sec)', 'accuracy(%)', 'precision', 'recall', 'F1', 'picks(%)', 'ms/trace'))
       else:
         lines = com.myprint(lines, '{0:12}{1:6}{2:>12}{3:>12}{4:>12}{5:>10}{6:>10}{7:>10}'.format('model', 'phase', 'mae (s)', 'accuracy(%)', 'precision', 'recall', 'F1', 'picks(%)'))
       lines = com.myprint(lines, '---------------------------------------------------------------------------------------')
       
       for phase in self.phases:
         if phase == 'p' or phase == 's':           
           for mode in self.picks_dict:
             #mae_true =list(filter(lambda x: x < 900, self.eval_dict[mode][phase]['mae']))
             mae_true =list(filter(lambda x: x > 0, self.eval_dict[mode][phase]['mae']))
             if len(mae_true)>0:
                mae = np.mean(mae_true)
             else:
                mae = -999
             precision = np.mean(self.eval_dict[mode][phase]['precision'])
             recall = np.mean(self.eval_dict[mode][phase]['recall'])
             if precision==0 and recall==0:
               f1=0
             else:
               f1 = 2 * ((precision * recall)/(precision + recall))
             acc = np.mean(self.eval_dict[mode][phase]['accuracy'])

             if self.time:
               avg_pre = 1000 *(np.mean(self.time_dict[mode]['pre'])/ np.mean(self.time_dict[mode]['count']))
               avg_eval = 1000 *(np.mean(self.time_dict[mode]['eval'])/np.mean(self.time_dict[mode]['count']))
               avg_post = 1000 *(np.mean(self.time_dict[mode]['post'])/np.mean(self.time_dict[mode]['count']))
               avg_time = avg_pre + avg_eval + avg_post
               
               lines = com.myprint(lines, '{0:<12}{1:<6}{2:>12.2f}{3:>12.2f}{4:>12.2f}{5:>10.2f}{6:>10.2f}{7:>10.2f}{8:>12.2f}'.format(mode, phase, 
                     mae , acc, precision, recall, f1, (self.eval_dict[mode][phase]['picks']*100/self.iter_no), avg_time))
             else:
               lines = com.myprint(lines, '{0:<12}{1:<6}{2:>12.2f}{3:>12.2f}{4:>12.2f}{5:>10.2f}{6:>10.2f}{7:>10.2f}'.format(mode, phase, 
                     mae , acc, precision, recall, f1, self.eval_dict[mode][phase]['picks']))
           lines = com.myprint(lines, '---------------------------------------------------------------------------------------') 
     return lines     
  
def main():
   """
   empty
   """
   test=Testbench(pick_modes=['caps'])
   
if __name__ == '__main__':
   main() 