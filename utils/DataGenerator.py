"""
A class to convert a SeedDataset object into processed data as CSV file
"""
import numpy as np
import obspy
from obspy.core.stream import Stream
import os
import sys
import random
import common as com
from SeedDataset import SeedDataSet
from obspy.signal.trigger import pk_baer, plot_trigger
import matplotlib.pyplot as plt
import pandas as pd

class DataGenerator:
   def __init__(self, dataset, channels=['HHZ'], f=100, filt=0, classes =['n', 'p', 's'],
                window=4, psize=5, ssize=5, npresize=40, npostsize=40, ptailsize=0, 
                stailsize=0, pslim=10, phase_start=0.5, numslices=40, streamproc='norm', sliceproc='stft', 
                prenorm=0, postnorm='max', dataset_type='train', train_test_ratio=0.9, seed=1,
                plot=0, shuffle=2, factor=2, nperf=1, num_plots=10, write_log=1, return_type='list', save_test =0):
     """
     dataset: SeedDataset object
 
     channels: seismic channels to select

     f: sampling rate for the generated data

     filt: denotes the filter to be used; 0: no filter; [x]: lowpass at x Hz; [x, y]: bandpass between x and y Hz

     classes: data classes

     window: size of data for each sample

     psize: seconds after arrival considered as p class

     ssize= seconds after arrival considered as s class

     npresize: seconds after start of trace considered as n class

     npostsize: seconds before end of trace considered as n class

     ptailsize: seconds after p considered as p tail, -1: not included, 0: include till next phase 

     stailsize: seconds after s considered as s tail, -1: not included, 0: include till next phase 
     
     pslim: minimum distance between P and S arrivals

     phase_start: time before a phase arrival to include in corresponding class

     numslices: number of slices to take from each class for one event

     streamproc: processing for the stream

     sliceproc: processing for the slice

     prenorm: normalization for actual data

     postnorm: normalization for processed data

     dataset_type: type of dataset, currently only csv is implemented

     train_test_ratio: The ratio of data to convert into training dataset

     plot: if plotting predefined number of data samples from (num_pots)

     shuffle: 0: no shuffle; 1: shuffle data slices 2: shuffle data slices as well as streams before train test division 

     half: 0: take full stft spectrum; 1:use only lower half

     num_plots: number of plots to make if plot=1

     write_log: if log is to be written

     return_type: the type of data to return (currently not implemented)

     save_test:
     """
     self.dataset = dataset
     self.ptime = dataset.ptime
     self.stime = dataset.stime
     self.channels = channels
     self.f =f
     self.filt=filt
     self.classes = classes # classes identified by the model
     self.num_classes = len(classes)
     self.window = window
     self.wlen = int(self.window * self.f)
     self.psize = psize # seconds after arrival considered as p class
     self.ssize = ssize # seconds after arrival considered as s class
     self.npresize = npresize # seconds after start considered as n class
     self.npostsize = npostsize # seconds before end considered as n class
     self.ptailsize = ptailsize # seconds after p considered as p tail, -1: not included, 0: include till next phase 
     self.stailsize = stailsize # seconds after s considered as s tail, -1: not included, 0: include till next phase 
     classes = ['p', 's', 'pre', 'post', 'ptail', 'stail']
     colors = ['r', 'b', 'g', 'g', 'pink', 'pink']
     self.class_colors={}
     for cl, color in zip(classes, colors):
       self.class_colors[cl] = color     
     self.pslim = pslim
     self.phase_start = phase_start # For p and s phases the earliest location of arrival in a window
     self.numslices=numslices # the number of slices per class from each event
     if sliceproc == 'stft':
       self.sliceproc=self.stft_proc
     elif sliceproc=='cwt':
       try:
         import mlpy
         self.sliceproc = com.get_cwt    
       except ModuleNotFoundError:
         import warnings
         warnings.warn("mlpy not installed, code snippet skipped")
         self.sliceproc = com.get_cwt2
     elif sliceproc == 'np':
       self.sliceproc=self.np_proc
     if streamproc == 'norm':
       self.streamproc = self.stream_norm
     self.return_type= return_type
     self.prenorm=prenorm
     self.postnorm = postnorm
     self.dataset_type= dataset_type
     self.ratio= train_test_ratio
     self.save_test= save_test
     self.seed= seed
     np.random.seed(self.seed)
     self.plot = plot
     self.shuffle = shuffle # 0: no shuffle; 1: shuffle data slices 2: shuffle data slices as well as streams before train test division 
     self.factor = factor # 
     self.nperf = nperf # 
     self.num_plots = num_plots
     self.num_streams = 0
     self.num_slices =0
     self.num_class_slices ={}
     for cl in self.classes:
       self.num_class_slices[cl]=0
     if write_log:
       import logging       
       logging.basicConfig(level=logging.INFO,
                format='%(levelname)s : %(asctime)s : %(message)s')
       self.logger = logging.getLogger()
     else:
       self.logger=0

   def npts(self, t):
     """
     This function obtains the location in a trace based on seconds from the trace start and self.f

     t: seconds from the trace start time or list of start times

     returns integer position in the trace as a single integer or list of positions defined by t
     """
     if isinstance(t, tuple) or isinstance(t, list):
       res = []
       for item in t:
         res.append(int(item* self.f))
     else:
         res = int(t* self.f)
     return res

   def get_name(self, string):
     """
     This function obtains a string from the defined parameters

     string: prefix for the name

     returns a string
     """
     if isinstance(self.filt, list):
       filt_str = f'_{self.filt[0]}_{self.filt[1]}' if len(self.filt)>1 else f'{self.filt[0]}'
     else:
       filt_str = f'{self.filt}'
     chan='_'.join(self.channels)
     print(chan)
     chan=chan.replace('?', 'all')
     chan=chan.replace('*', 'all')
     print(chan)
     name = f'{string}_{self.dataset_type}_{self.f}hz_{filt_str}filt_{self.window}s_{chan}channels'
     if (self.factor) : name+=f'_factor{self.factor}'
     if (self.nperf) : name+=f'_nperf{self.nperf}'
     if (self.ptailsize>=0) or (self.stailsize>=0) : name+='_tails'
     if self.prenorm!=0:
       name+=f'_prenorm_{self.prenorm}'
     if self.postnorm!=0:
       name+=f'_postnorm_{self.postnorm}'
     return name
  
   def get_arrivals_npts(self, st_npts):
     """
     This function computes the location of P and S arrivals based on npts of current trace

     st_npts: total points in the trace

     returns a dictionary with phase arrivals as npts
     """        
     arrivals={}
     arrivals['p']= self.npts(self.ptime) # location of p arrival
     arrivals['s']= st_npts-self.npts(self.stime) # location of s arrival
     return arrivals

   def get_arrivals(self, st_len):
     """
     This function computes the time of P and S arrivals based on length of current trace in seconds

     st_npts: len of trace

     returns a dictionary with phase arrivals in seconds
     """        
     # location of phase arrivals          
     arrivals={}
     arrivals['p']= self.ptime # location of p arrival
     arrivals['s']= st_len-self.stime # location of s arrival
     return arrivals

   def get_boundaries_npts(self, arrivals, st_npts):
     """
     This function computes the boundaries of different phases segments of the traces based on P and S arrivals and npts of current trace. This function segregates noise taken from start and end of trace and tails taken from each phase in order to ensure equal representation of all regions included in a class.

     arrivals: dictionary with phase arrivals as npts

     st_npts: total points in the trace

     returns a dictionary with phase boundaries in num points (npts)
     """        
     boundaries={}
     p_len = self.npts(self.psize) # p phase width
     s_len = self.npts(self.ssize) # s phase width        
     phase_offset = self.npts(self.phase_start) # number of points taken before phase arrival
     boundaries['p'] =(arrivals['p']- phase_offset , arrivals['p'] + p_len+ phase_offset ) # boundaries of p class
     boundaries['s'] = (arrivals['s']- phase_offset , arrivals['s'] + s_len+ phase_offset) # boundaries of s class
     boundaries['pre'] = (0, min(self.npts(self.npresize), arrivals['p'])) # boundaries of noise class taken from start
     boundaries['post'] = (max(st_npts - self.npts(self.npostsize), boundaries['s'][1]), st_npts) # boundaries of noise class taken from end
     if not self.ptailsize < 0:
       if self.ptailsize == 0:
         boundaries['ptail'] = (boundaries['p'][1], arrivals['s'])
       else:
         boundaries['ptail'] = (boundaries['p'], min(self.npts(self.ptailsize), arrivals['s']))
     if not self.stailsize < 0:       
       if self.stailsize == 0:
         boundaries['stail'] = (boundaries['s'][1], boundaries['post'][0])
       else:
         boundaries['stail'] = (boundaries['s'][1], min(self.npts(self.stailsize), boundaries['post'][0]))
     return boundaries 

   def get_boundaries(self, arrivals, st_len):
     """
     This function computes the boundaries of different phases segments of the traces based on P and S arrivals and len of current trace in seconds. This function segregates noise taken from start and end of trace and tails taken from each phase in order to ensure equal representation of all regions included in a class.

     arrivals: dictionary with phase arrivals as seconds

     st_npts: total time of the trace

     returns a dictionary with phase boundaries in seconds
     """    
     boundaries={}
     p_len = self.psize # p phase width
     s_len = self.ssize # s phase width        
     phase_offset = self.phase_start # number of points taken before phase arrival
     boundaries['p'] =(arrivals['p']- phase_offset , arrivals['p'] + p_len+ phase_offset ) # boundaries of p class
     boundaries['s'] = (arrivals['s']- phase_offset , arrivals['s'] + s_len+ phase_offset) # boundaries of s class
     boundaries['pre'] = (0, min(self.npresize, arrivals['p'])) # boundaries of noise class taken from start
     boundaries['post'] = (max(st_len - self.npostsize, boundaries['s'][1]), st_len) # boundaries of noise class taken from end
     if not self.ptailsize < 0:
       if self.ptailsize == 0:
         boundaries['ptail'] = (boundaries['p'][1], arrivals['s'])
       else:
         boundaries['ptail'] = (boundaries['p'], min(self.ptailsize, arrivals['s']))
     if not self.stailsize < 0:       
       if self.stailsize == 0:
         boundaries['stail'] = (boundaries['s'][1], boundaries['post'][0])
       else:
         boundaries['stail'] = (boundaries['s'][1], min(self.stailsize, boundaries['post'][0]))
     return boundaries 

   def get_class_fill(self, arrivals, boundaries):      
      """
      This function uploads the arrivals and phase boundaries according to the class as assigns a color to each class according to self.class_colors

      arrivals: dictionary with phase arrivals in seconds

      boundaries: dictionary with phase boundaries in seconds

      returns a dictionary with phase arrivals and boundaries in num points (npts) as well as assigned colors for plotting
      """    
      class_fill ={}
      #for key in boundaries: print(key)
      for i, cl in enumerate(self.class_colors):
        class_fill[cl]={}
        class_fill[cl]['color'] =self.class_colors[cl]
        if (cl=='p') or (cl=='s'):
          class_fill[cl]['arrival']=self.npts(arrivals[cl])
          class_fill[cl]['boundaries']=self.npts(boundaries[cl])
        else:
          class_fill[cl]['arrival']=0              
          if cl=='pre' or cl=='post' : class_fill[cl]['boundaries']=self.npts(boundaries[cl])
          elif cl=='ptail' and self.ptailsize >=0: class_fill[cl]['boundaries']=self.npts(boundaries[cl])
          elif cl=='stail' and self.stailsize >=0: class_fill[cl]['boundaries']=self.npts(boundaries[cl])
          else: class_fill[cl]['boundaries']=(0, 0)
      return class_fill 

   def get_rand_start(self, data, num):
      """
      This function provides a given number of random starting points for a given array and class parameters

      data: input data as numpy array

      num: number of starting points to provide

      returns two lists: one of integers indicating random starting points and the seconds of corressponding data slices
      """
      starts = np.random.randint(0, len(data)-self.wlen, num).tolist()
      slices = []
      for i in np.arange(len(starts)):
        sl = data[starts[i]:(starts[i]+self.wlen)]
        slices.append(sl)
      return slices, starts  

   def get_rand_start_st(self, st, num):
      """
      This function provides a given number of random starting points for a stream and class parameters

      data: input data as stream

      num: number of starting points to provide

      returns two lists: one of integers indicating random starting points and the seconds of corressponding data sliced streams
      """
      info = com.get_info(st)
      len_st = info['endt'] - info['stt']
      start_ind = np.random.uniform(0, len_st-self.window, num).tolist()
      slices = []
      starts = []
      for i in np.arange(len(start_ind)):
        start = info['stt']+start_ind[i] 
        sl = st.slice(start, start+self.window)
        slices.append(sl)
        starts.append(start)
      return slices, starts              

   def plot_class_boundries(self, stream, n, arrivals, boundaries, plotpath):
     """
     This functions generates a plot superimposing class on the given stream

     stream: obspy stream object

     n: stream number

     arrivals: dict with arrivals in npts

     boundaries: dict with class boundaries in npts

     plotpath: directory where plot is to be saved
     """
     class_fill = self.get_class_fill(arrivals, boundaries)     
     alpha = 0.4
     if len(self.channels)>1:
       fig, axes = plt.subplots(len(self.channels), 1, figsize=(9,8), sharex=True)
       for i, ax in enumerate(axes):
         data = stream[i].data
         ax.plot(data)         
         ymin = min(data)
         ymax = max(data)
         ylim = com.get_lim(ymin, ymax, 0.2)
         for cl in class_fill:
           ax.axvspan(class_fill[cl]['boundaries'][0], class_fill[cl]['boundaries'][1], alpha=alpha, color=class_fill[cl]['color'], label =f'{cl}')
         if class_fill[cl]['arrival']>0:
           ax.vlines(class_fill[cl]['arrival'], ymin = 1.2 * ymin, ymax=1.2 * ymax, color=class_fill[cl]['color'], label =f'{cl} arrival')
         ax.set_ylabel(stream[i].stats.channel)
         ax.set_ylim(1.2 * ymin, 1.2 * ymax)
         ax.legend()
       ax.set_xlabel('time [sec ]')    
       ax.set_xlim(0, 1.2 * len(data))    
     else:
       data = stream.data
       plt.figure(figsize=(9,3))
       alpha = 0.4
       plt.plot(data)
       ymin = min(data)
       ymax = max(data)
       ylim = com.get_lim(ymin, ymax, 0.2)
       for cl in class_fill:
         plt.axvspan(class_fill[cl]['boundaries'][0], class_fill[cl]['boundaries'][1], alpha=alpha, color=class_fill[cl]['color'], label =f'{cl}')
         if class_fill[cl]['arrival']>0:
           plt.vlines(class_fill[cl]['arrival'], ymin = 1.2 * ymin, ymax=1.2 * ymax, color=class_fill[cl]['color'], label =f'{cl} arrival')
       plt.xlabel('time [sec * f]')
       plt.ylabel(stream.stats.channel)
       plt.ylim(ylim[0], ylim[1])
       plt.xlim(0, 1.2 * len(data))
       plt.legend()
     plt.tight_layout()     
     plt.savefig(os.path.join(plotpath, f'st_{n}_classes.png'))
     plt.close()
   
   # stream preprocessed for STFT based approach
   def stft_proc(self, stream, to_numpy=1):
     '''
     This function processes a stream by taking STFT

     stream: stream to be processed

     to_numpy: if output should be converted to ndarray

     returns the processed data     
     '''
     norm= 1 if self.postnorm=='max'else 0
     
     return com.stft_proc(stream, norm=norm, f=self.f, filt=self.filt, channels=self.channels, factor=self.factor, nperf=self.nperf, to_numpy=1)

   # stream preprocessed for STFT based approach
   def np_proc(self, stream, to_numpy=1, squeeze=1):
     '''
     This function processes a stream by taking STFT

     stream: stream to be processed

     to_numpy: if output should be converted to ndarray

     returns the processed data     
     '''
     norm= 1 if self.postnorm=='max'else 0 
     data = com.get_numpy_t(stream)
     
     if squeeze: data = np.squeeze(data)
          
     data = com.normalize(data, norm=self.postnorm)
     
     return data

   def stream_norm(self, st, f_st):
     """
     This function performs preprocessing for the entire stream including sampling rate adjustment and if stream-wise normalization is used then it is also normalized

     st: obspy stream

     f_st: sampling rate of the stream

     returns processed stream
     """       
     if f_st != self.f:
       print(f'Stream interpolated from frequency {f_st} to {self.f}.')     
       st.interpolate(sampling_rate=self.f)
     if self.prenorm=='local':         
       st=st.normalize() # use a normalization function for stream here
     return st

   def shuffle_dict(self, data_dict):
     """
     This function shuffles the data dictionary

     data_dict: dictionary with data
     """
     #data_dict['data'] = np.array(data_dict['data'])     
     sl_count = len(data_dict['data']) 
     rand_ind = np.random.permutation(sl_count)
     shuffled_dict ={}
     for key in data_dict:
       try:
         shuffled_dict[key]=np.array(data_dict[key])[rand_ind]
       except:
         print('Failed to shuffle')
         for item in data_dict[key]:
           if item.shape != (26, 9): 
             shape= item.shape         
         return data_dict            
     return shuffled_dict    
        

   def get_hist(self, data_dict):
      """
      This function generates a weighted histogram of data wrt to frequency spectrum

      data_dict: data dictionary

      returns a dictionary with information required to plot a histogram
      """
      my_hist = {}
      sl_count = len(data_dict['data']) 
      for cnt in np.arange(sl_count):
        phase = data_dict['phase'][cnt]
        if phase not in my_hist:
          my_hist[phase] = {}
          my_hist[phase]['sum']=0
          my_hist[phase]['count']=0
        data = np.array(data_dict['data'][cnt])          
        for i in range(1, data.ndim):
          data = np.sum(data, axis=-1)       
        my_hist[phase]['sum']+=data
        my_hist[phase]['count']+=1
      for phase in my_hist:
        my_hist[phase]['norm'] = my_hist[phase]['sum']/my_hist[phase]['count']
      return my_hist
    
   def plot_hist(self, my_hist, outdir): 
      """
      This function plots the data histogram and saves it to given directory

      my_hist: dictionary with weighted histogram data

      outdir: directory where plot is to be saved
      """          
      plt.figure()        
      for phase in self.classes: 
        count = my_hist[phase]['count']        
        freqs = np.arange(my_hist[phase]['sum'].shape[0])       
        plt.plot(freqs,  my_hist[phase]['norm'], label = f'{phase} phase ({count} events)')
        
      plt.xlabel('Freq [Hz]')
      plt.ylabel('Power')
      #plt.ylim(0, 1.1* ymax)
      plt.tight_layout() 
      plt.legend()
      plt.savefig(os.path.join(outdir, 'fft_hist.png'))
      plt.close() 

   # make all the plots
   def plot_transform_2d(self, data, fs, trans, sta, stt, endt, chn, eventdir, label, fname, markers = [], mlabels = []):
     # waveforms
     data_len = len(data)
     data_axis = np.arange(0, (data_len/fs), 1/fs)
     dels = [0.2, 0.1, 0.1, 0.1]
     colors = ['r', 'b', 'g']
     ymax = max(data)
     ymin = min(data)
     fig, ax = plt.subplots(2, 1)
     data_axis=data_axis[:data_len]
     
     #exit()
     # plotting the signal 
     ax[0].plot(data_axis, data, color ='green', label=chn)
     if len(markers) > 0:
       ax[0].set_ylim(ymin-0.3, ymax+0.3)
       for m, marker in enumerate(markers):
         ax[0].vlines(marker/fs, ymin = -dels[m] + ymin, ymax=dels[m] + ymax, 
                                   color=colors[m], label = mlabels[m])
         ax[0].scatter([marker/fs, marker/fs], [-dels[m]+ ymin, dels[m]+ ymax], marker ='_', color=colors[m])
     ax[0].set_xlabel('Time [sec]')
     ax[0].set_ylabel('Amplitude')
     ax[0].set_title(label + f': sta={sta} stt={stt.day}T{stt.hour}:{stt.minute}:{stt.second}')
     
     # plotting the magnitude spectrum of the signal 
     Zxx = trans
     t=np.arange(trans.shape[1])
     f = np.arange(trans.shape[0])
     if len(markers) > 0:
       #print(len(data), Zxx.shape, t.shape)
       ymin, ymax = np.amin(f), np.amax(f)
       ax[1].set_ylim(ymin-0.3, ymax+0.3)
       for m, marker in enumerate(markers):
         ax[1].vlines(marker/fs, ymin = -dels[m] + ymin, ymax=dels[m] + ymax, 
                                   color=colors[m], label = mlabels[m])
         ax[1].scatter([marker/fs, marker/fs], [-dels[m]+ ymin, dels[m]+ ymax], marker ='_', color=colors[m])
     im = ax[1].pcolormesh(t, f, Zxx, shading ='auto')
     ax[1].set_xlabel('Time [sec]')
     ax[1].set_ylabel('Frequency [Hz]')
           
     #plt.legend()
     plt.colorbar(im)
     plt.tight_layout()  
     plt.savefig(os.path.join(eventdir, fname+'.png'))  
     plt.close()


   def generate_dataset(self, outdir):
      """
      This function generates a data set and saves it to a directory

      outdir: directory for dataset
      """  
      stream_list = self.dataset.stream_to_list(f=[self.f], channels=self.channels, norm=self.prenorm)
      
      if len(stream_list) == 0:
        print(f'No events were found in {self.dataset.path} for {self.f}Hz and channels={self.channels}')
        print(self.dataset.channels)
        print(self.dataset.freq)
        exit(0)
      com.safe_mkdir(outdir)        
      print(f'{outdir} folder is created.')
      if self.plot==2:
        eventspath = os.path.join(outdir, 'events') #
        com.safe_mkdir(eventspath)
        print(f'{eventspath} folder is created.')
      if self.shuffle==2:
         random.seed(self.seed)
         stream_list = random.sample(stream_list, len(stream_list))
         print('The stream is shuffled.')
      data_len= self.npts(self.window)
      fft_len = 1 + (data_len//2) # length of fft
      self.len = len(stream_list)
      if self.save_test: self.train_len = int(self.ratio * self.len)
      else: self.train_len = self.len
      self.test_len = self.len - self.train_len
      st_list = stream_list[:self.train_len] if self.dataset_type=='train' else stream_list[self.train_len:]
      if self.save_test:
        testpath = os.path.join(outdir, 'test_traces') #
        com.safe_mkdir(testpath)        
        st_test = stream_list[self.train_len:]        
        for st in st_test:
          com.save_stream(st, testpath)
        print(f'{len(st_test)} traces are saved to the {testpath} folder.')
      data_dict = {}
      data_dict['stream_id'] =[]  
      data_dict['slice_start'] = []
      data_dict['sampling_rate'] =[]
      data_dict['channel'] =[]
      data_dict['phase'] = []
      data_dict['data'] = [] 
        
        
      # Loop through list
      for n, st in enumerate(st_list[::]):
        remove_tr = 0
        if self.plot==2 and n<self.num_plots:
          eventdir = os.path.join(eventspath, f'event{n}')
          com.safe_mkdir(eventdir)    
        if self.logger: self.logger.info(f"Processing trace : {n}/{len(st_list)}")
        info = com.get_info(st)
        f_st = info['f']
          
        st = self.streamproc(st, f_st)
        
        # trace statistics
        stt = info['stt']
        endt = info['endt']
        sta = info['sta']
        net = info['net']
        loc = info['loc']
        st_npts = (info['npts']* self.f)/f_st
        # id for a trace
        id = com.get_id(st)
        # get class divisons          
        arrivals=self.get_arrivals(endt-stt)
        boundaries=self.get_boundaries(arrivals, endt-stt) 
        
        for key in boundaries:
          if boundaries[key][1] - boundaries[key][0] < (self.window*(1+ self.phase_start)):
            print(f'The required amount of data for phase {key} is not available thus the stream {n} is dropped')
            remove_tr = 1
            continue
        if len(st.data)< self.f*(endt-stt):
          print('gap detected thus trace removed', len(st.data), self.f*(endt-stt))
          continue
        else:
          st.data = st.data[:int(self.f*(endt-stt))]
        if remove_tr: continue
        if (n==0):
          class_keys = {}
          for key in self.classes:
            if key in boundaries:
              class_keys[key]=[key]
            else:
              class_keys[key]=[]
              for new_key in boundaries:
                if new_key not in self.classes:
                  if key == 'tail' and new_key in ['ptail', 'stail']:
                   class_keys[key].append(new_key)
                  elif key == 'n':
                   class_keys[key].append(new_key)
                            
          if self.plot==1: self.plot_class_boundries(st, n, arrivals, boundaries, outdir) 
          
        slice_dict = {}        
          
        for key in class_keys:
          slice_dict[key]={}
          slice_dict[key]['slices'], slice_dict[key]['starts']=[], []
          for data_key in class_keys[key]:
            #print(key, data_key, boundaries[data_key][0], boundaries[data_key][1])
            data=st.slice(stt+boundaries[data_key][0], stt+boundaries[data_key][1])        
            num =  self.numslices//len(class_keys[key])     
            slices, starts= self.get_rand_start_st(data, num)
            slice_dict[key]['slices']+=slices
            slice_dict[key]['starts']+=starts
        flag = 0      
        for key in slice_dict:
          for sl_ind, slice in enumerate(slice_dict[key]['slices']):
            proc_data = self.sliceproc(slice)  
            if flag==0:
              proc_shape = proc_data.shape
              flag= 1
            if proc_data.shape!=proc_shape:
              print(f'slice of shape {proc_data.shape} is skipped')
              continue
            data_dict['stream_id'].append(id)
            data_dict['slice_start'].append(slice_dict[key]['starts'][sl_ind])
            data_dict['sampling_rate'].append(self.f)
            data_dict['channel'].append('/'.join(self.channels))
            data_dict['phase'].append(key)
            #proc_data = self.sliceproc(slice)      
            data_dict['data'].append(proc_data) 
            self.num_slices+=1 
            self.num_class_slices[key]+=1   
            if self.plot==2: self.plot_transform_2d(slice.data, self.f, proc_data, sta, stt, endt, self.channels, eventdir, 'stft', fname=f'{key}_slice{sl_ind}')          
        self.num_streams+=1            
        
      if self.shuffle: 
        data_dict= self.shuffle_dict(data_dict)
        
      if self.plot:
        my_hist = self.get_hist(data_dict)
        self.plot_hist(my_hist, outdir)
      data_dict = com.dict_for_pd(data_dict)
      df = pd.DataFrame.from_dict(data_dict)
      df.to_csv(os.path.join(outdir, 'dataset.csv'))
      
def main1():
   """
   A working example for using the class

   # Define data path
   dataPath = "..\\SeedData\\ISBData2018_2021\\*\\*.mseed"

   # define channels and frequency to select

   channels=['HHZ']

   f=100

   # if dataset for training or test is to be generated

   dataset_type='train'   

   # define SeedDataSet object based on data path and phase arrival times
 
   dataset = SeedDataSet(dataPath, ptime=60, stime=200) 

   # define DataGenerator object for the dataset with required params
     
   gendata = DataGenerator(dataset, dataset_type=dataset_type, channels=channels, plot=1, filt=[5])

   # ouput path
   basepath = f'..\\CSV_datasets\\{gendata.dataset_type}_dataset'

   datadir= os.path.join(basepath, gendata.get_name('test_Peshwar_stft'))

   # generate dataset in the output path given

   gendata.generate_dataset(datadir)

   # get configuration of data generation

   config = com.get_class_config(gendata, [com.decorate('Dataset Configuration')]) 

   com.to_file(config, os.path.join(datadir, 'config.txt'))
   """
   peshPath = "..\\SeedData\\PESH2016_2019\\train\\*.mseed" 
   SeedPath = "..\\SeedData\\IRIS_2018_2020\\train\\*.mseed"
   isbPath = "..\\SeedData\\ISB_2018_2021\\train\\*.mseed"
   name = 'IRIS'
   if name== 'ISB': dataPath = isbPath
   elif name== 'PESH': dataPath = peshPath
   elif name== 'IRIS': dataPath = SeedPath

   #channels=['HHE', 'HHN', 'HHZ']
   channels=['BH?']
   #channels=['BHZ']
   f=40
   ptime =60
   if name== 'ISB': stime = 120
   elif name== 'PESH': stime = 200
   elif name== 'IRIS': stime = 300
   
   dataset_type='train'    
   dataset = SeedDataSet(dataPath, ptime=ptime, stime=stime)      
   gendata = DataGenerator(dataset, dataset_type=dataset_type, channels=channels, f=f, plot=1, filt=0, factor=2)
   basepath = f'..\\CSV_datasets\\{gendata.dataset_type}\\{name}'
   datadir= os.path.join(basepath, gendata.get_name(name))
   gendata.generate_dataset(datadir)
   config = com.get_class_config(gendata, [com.decorate('Dataset Configuration')]) 
   com.to_file(config, os.path.join(datadir, 'config.txt'))

def main2():
   peshPath = "..\\SeedData\\PESH2016_2019\\train\\*.mseed" 
   SeedPath = "..\\SeedData\\IRIS_2018_2020\\train\\*.mseed"
   isbPath = "..\\SeedData\\ISB_2018_2021\\train\\*.mseed"
   name = 'IRIS'
   if name== 'ISB': dataPath = isbPath
   elif name== 'PESH': dataPath = peshPath
   elif name== 'IRIS': dataPath = SeedPath

   #channels=['HHE', 'HHN', 'HHZ']
   #channels=['HH?']
   #channels=['BHZ']
   f=40
   ptime =60
   if name== 'ISB': stime = 120
   elif name== 'PESH': stime = 200
   elif name== 'IRIS': stime = 300
   
   dataset_type='train'    
   dataset = SeedDataSet(dataPath, ptime=ptime, stime=stime)      
   gendata = DataGenerator(dataset, dataset_type=dataset_type, channels=channels, f=f, filt=[5], plot=1, prenorm=0, postnorm='unity', sliceproc='np')
   basepath = f'..\\CSV_datasets\\{gendata.dataset_type}\\{name}'
   datadir= os.path.join(basepath, gendata.get_name(name))
   gendata.generate_dataset(datadir)
   config = com.get_class_config(gendata, [com.decorate('Dataset Configuration')]) 
   com.to_file(config, os.path.join(datadir, 'config.txt'))

if __name__ == '__main__':
   main1() 