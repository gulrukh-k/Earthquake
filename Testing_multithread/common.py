"""
A collection of the utility functions required by the various modules
"""
import os
import numpy as np
import obspy
from collections import Counter
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
import tensorflow
# print to screen and append to list
def myprint(lines, line=[], to_print=1):
   '''
   lines: list to which the text should be appended
   line: text to display and add to list
   '''
   if to_print: print(line)
   lines.append(line+'\n')
   return lines


def safe_mkdir(path):
   '''
   This function creates a directory
   Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)

   path: folder to create
   '''
   from os import makedirs
   from errno import EEXIST
   try:
       makedirs(path)
   except OSError as exception:
       if exception.errno != EEXIST:
           raise exception

def get_loc(t, st, fr):
   '''
   This function provides the locus in a trace corresponding to time

   t: time to convert

   st: trace starting time

   ft: sampling rate

   returns location in the trace
   '''
   return (t-st)* fr

def fill_empty(ls):
   '''
   This function fills empty list with 0.0

   ls: empty list []

   returns [0.0]
   '''
   if len(ls)==0:
     ls =[0.0]
   return ls

def normalize(data, norm='max', max=0, dmin=0, dmax=0):
   '''
   This function normalizes the data in different formats

   data: data to be normalized

   norm: the type of normalization; 'abs1': from -1 to 1; 'unity': from 0 to 1; 'max': diving by max; 'stat': statistical resulting in 0 mean and 1 std

   max: the max value to use for max normalization; 0: calculate from data

   dmin: data minimum (0 indicates to compute from data)

   dmax: data maximum (0 indicates to compute from data)

   returns normalized data
   '''
   if norm == 'abs1':
     if dmin==0 and dmax==0:
       norm = ((np.max(data)-np.min(data))*(data+1))/(2*np.min(data))
     else:
       norm = ((dmax-dmin)*(data+1))/(2*dmin)
   elif norm == 'unity':
     if dmin==0 and dmax==0:
       norm = (data -np.min(data))/(np.max(data)-np.min(data))
     else:
       norm = (data -dmin)/(dmax-dmin)
   elif norm == 'max':
     if max==0:
       max = np.max(np.absolute(data))
     norm = data/max
   elif norm == 'stat':
     norm = (data - np.mean(data)) / np.std(data)
   return norm

def inv_norm(data, norm='max', max=0, dmin=0, dmax=0):
   '''
   This function inverses the normalization of the data in different formats

   data: input data

   norm: the type of normalization; 'abs1': from -1 to 1; 'unity': from 0 to 1; 'max': diving by max; 'stat': statistical resulting in 0 mean and 1 std

   max: the max value to use for max normalization; 0: calculate from data

   dmin: data minimum (0 indicates to compute from data)

   dmax: data maximum (0 indicates to compute from data)

   returns normalized data
   '''
   if norm == 'abs1':
     if dmin==0 and dmax==0:
       inv = 1 - ((np.max(data)-np.min(data))/(2 * (data -np.min(data))))
     else:
       inv = 1 - ((dmax-dmin)/(2 * (data -dmin)))
   elif norm == 'unity':
     if dmin==0 and dmax==0:
       inv = dmin + (data *(np.max(data)-np.min(data)))
     else:
       inv = dmin + (data * (dmax-dmin))
   elif norm == 'max':
     if max==0:
       max = np.max(np.absolute(data))
     inv = data*max
   elif norm == 'stat':
     inv = (data * np.std(data)) + np.mean(data) 
   return inv

def most_frequent(list):
    """
    This function obtains the most frequenctly represented value in the list

    list: the input list

    returns the most frequenct value
    """
    occurence_count = Counter(list)
    return occurence_count.most_common(1)[0][0]

def dict_for_pd(data_dict):
    """
    This function converts data dictionary to a format suitable for conversion into pandas

    data_dict: original dictionary with data

    return modified dictionary
    """
    new_dict={}
    for key in data_dict:
      if isinstance(data_dict[key][0], (np.ndarray)): 
        temp = process_for_pd(data_dict[key])
        for temp_key in temp:
          new_dict['data_'+temp_key] = temp[temp_key]
      else:
        new_dict[key]=data_dict[key]
    return new_dict

def process_for_pd(array):
   """
   This function processes a multi-dimensional array into a dictionary with 1d arrays

   array: multi-dimensional numpy array

   return a dictionary for each array dimension
   """
   array = np.array(array) # in case input is a list
   new_dict={}         
   for indices in product(*map(range, array.shape[1:])):
      key = ''
      slc = [slice(None)]
      for ind in indices:
        key+=f'{ind}_'
        slc.append(slice(ind, ind+1, 1))
      new_array = array[tuple(slc)]
      new_dict[key[:-1]] = np.squeeze(new_array) # slicing creates redundant dims
   return new_dict

def get_stft(data, f, norm = 1, half = 1):
   """
   This function obtains the stft of seismic data

   f: sampling frequency of data

   norm: if normalizing the spectrum

   half: if half of the spectrum is to be kept
   """ 
   from scipy import signal
   data_len = len(data)
   f, t, Zxx = signal.stft(data, f, nperseg=f, noverlap=f//2, window=('hamming'))
   Zxx = np.abs(Zxx)
   if norm: Zxx = Zxx/np.amax(Zxx)
   if half:
     size = len(f)/2
     for i in range(half-1):
       size = size/2
     size = int(size) + 1 
     f = f[:size]
     Zxx = Zxx[:size, :]
   return  Zxx

def subtract_str(str1, str2):
   """
   This function subtracts a substring from a string

   str1: given string

   str2: substring to be subtracted

   returns difference
   """
   num = len(str2)
   if str2 in str1:
     for i in range(len(str1)-num):
       if str1[i:i+num]==str2:
         ind = i
         continue
     return str1[:i+1] + str1[i+num+1:]
   return str1

def get_class_config(myclass, config=[]):
   """
   This function obtains the parameters values from a class as text and can also append it to provided text.

   myclass: the class whose configuration is required

   config: previous text to append the output with

   reurns the text with class configuration
   """
   import inspect
   # members of an object 
   for i in inspect.getmembers(myclass):      
     # to remove private and protected
     # functions
     if not (i[0].startswith('_') or i[0]=='stream'):          
        # To remove other methods that
        # doesnot start with a underscore
        if not (inspect.ismethod(i[1]) or hasattr(i[1], '__call__')): 
            line = f'{i[0]}: '
            if type(i[1])==list: 
              if len(i[1])>0:
                if isinstance(i[1][0], tensorflow.keras.callbacks.EarlyStopping):             
                  for item in i[1]:                
                    label= str(type(item))
                    line = line + label
                else:                
                  line = line + f'{i[1]}'
              line = line + '\n'
           
            elif type(i[1])==dict:
              for key, value in i[1].items():
                if type(value)==dict:
                   line = line +f'{key}:{value.keys()}  '
                else: 
                   line = line +f'{key}:{value}  '           
              line = line +'\n'
            elif hasattr(i[1], 'name'):
              line = line + i[1].name + '\n'
            elif hasattr(i[1], 'shape'):
                line = line +  f' array of shape: {i[1].shape}\n' 
            else:                         
              line = line + f'{i[1]}  \n'              
            config.append(line)
   config.append('\n')
   return config

def to_file(text, txtfile=''):
   """
   This function writes text to a file.

   text: text to be written

   txtfile: file name
   """
   with open(txtfile, 'w') as file:
       file.writelines(text) 

def print_dict(dictionary, ident = '', braces=1, line=' '):
    """ 
    This function recursively prints nested dictionaries.

    ident, braces and line are required for recursive application 

    returns text
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            line = line + f'{ident} [{key}]\n'
            line = print_dict(value, ident+'  ', braces+1, line)
        #else:
        #   line = line + ident+f'{key}\n'
    return line

def decorate(line, size = 30, shape='*'):
   """
   This function encloses some text using lines of a certain shape. This is useful for documentation

   line: line to be decorated

   size: size of the decoration

   shape: shape to be used

   returns the input text enclosed by given shapes
   """
   lines = ''   
   for i in range(size):
     lines= lines + shape
   lines = lines + f'\n{line}\n'
   for i in range(size):
     lines = lines + shape
   lines = lines + '\n'
   return lines

def remove_spaces(string):
    """
    This function removes spaces from a string
   
    returns string with spaces removed
    """  
    return "".join(string.split())

# getting stats for STA LTA based trigger       
def get_stalta_stats(a, hist_avg, f, sta_win=1, iter_count=0):
   """
   This function the stats needed for implementation of sta/lta trigger

   a : data

   hist_avg: historical average

   f: frequency

   sta_win: window size for sta

   iter_count: iteration count

   returns sta, lta and hist_avg
   """
   lta = np.mean(a ** 2)
   sta = np.mean(a[-int(sta_win * f):] ** 2)  
   lta = np.mean(lta)
   sta = np.mean(sta)
   if iter_count == 0:
     hist_avg = lta
   else:
     hist_avg = ((iter_count-1) * hist_avg + lta)/iter_count
   return sta, lta, hist_avg

def text_to_file(text, filename):
   """
   This function writes text to a file.

   text: text to be written

  filename: file name
   """
   with open(filename, 'w') as file:
        file.writelines(text) 

# make title based on seismic event info
def make_title_info(info, snr=None):
   """
   This function makes a title based on seismic event info and snr

   info: info from stream

   snr : snr from event
  
   returns title string
   """
   date = date_format(info['stt'])
   start = time_format(info['stt'])
   end =   time_format(info['endt'])
   f = info['f']
   sta = info['sta']
   label = f'Station={sta} {date}:  {start} to {end} [{f}Hz]'
   if snr: label = label + f' snr: {snr:.2f}'  
   return label

def obspy_utils():
   """
   ---------------------------------------------- OBSPY utils ------------------------------------------------------------

   The following are functions performed on seismic data requiring obspy
   """

def npts(t, f=0):
     """
     This function obtains the location in a trace based on seconds from the trace start and self.f

     t: seconds from the trace start time or list of start times

     returns integer position in the trace as a single integer or list of positions defined by t
     """
     if f==0: f = self.f
     if isinstance(t, tuple) or isinstance(t, list):
       res = []
       for item in t:
         res.append(int(item* f))
     else:
         res = int(t* f)
     return res
  
# stream preprocessed for STFT based approach
def stft_proc(stream, norm=1, f=None, filt=0, channels=[], half=0, to_numpy=1, squeeze=1):
   '''
   This function procesess a stream by taking STFT

   stream: stream to be processed

   norm: if data is to be normalized

   f: frequency

   filt: if data is to be filtered

   channels: channels to select

   half: if half of the spectrun is to be obtaine

   to_numpy:  if data has to be converted in an array

   squeeze: if redundant dimensions are to be removed

   returns an array of processed data     
   '''
   st = stream.copy()    # make a copy of stream for preprocessing
   st.detrend('demean')  # remove mean
   
   info=get_info(stream)
   if f and (info['f'] != f): # if original frequency is different then interpolate to required frequency
     print('Stream interpolated from frequency {} to {}.'.format(info['f'], f))  
     st.interpolate(sampling_rate=f)  
   if filt: # apply filter if required
     
     if len(filt)==1:
       st.filter('lowpass', freq=filt[0])
     else: 
       st.filter('bandpass', freqmin=filt[0], freqmax=filt[1])  # optional prefiltering  
     
   # get the transform as list
   stft_data = []
   if len(channels) > 1:
     for count, ch in enumerate(channels):
       if ch[-1] in ['*', '?']:
         ch = ch[-1] 
       for tr in st:
         if (ch in tr.stats.channel) and (len(stft_data) < len(channels)):
           stft_data.append(get_stft(tr.data[:-1], f, norm=norm, half=half))
   elif len(channels) == 1:
     if hasattr(st, 'stats'): st=[st]
     for tr in st:
       #tr=st
       ch = channels[0]
       if ch[-1] in ['*', '?']:
         ch = ch[:-1]  
       if ch in tr.stats.channel:
         stft_data.append(get_stft(tr.data[:-1], f, norm=norm, half=half))
   else:
     for tr in st:
       stft_data.append(com.get_stft(tr.data[:-1], f, norm=norm, half=half))
   
   if to_numpy:
     # reshape to required format
     stft_data = get_numpy(stft_data)
     stft_data = np.moveaxis(stft_data, 1, -1)
     if squeeze: stft_data = np.squeeze(stft_data)
   return stft_data

def get_numpy_t(st):
   """
   This function converts an obspy stream object into a transposed numpy array

   st: stream object

   returns an array
   """
   return np.array([i.data for i in st]).T[np.newaxis, ...]
   
def get_numpy(st):
   """
   This function converts an obspy stream object into a numpy array

   st: stream object

   returns an array
   """
   return np.array([i.data for i in st])[np.newaxis, ...]

def get_info(data, mode='single'):
   """
   This function obtains info related to a stream, trace or list of streams
   
   data : input data

   mode : 'single' if the data of only the first element in a list of traces is required.
    if mode is not single then the info from all traces is obtained.

   returns stat info for seismic data.
   """   
   info = {}
   if hasattr(data, 'stats'):
     # get event stats from trace
     info['stt'] = data.stats.starttime
     info['endt'] = data.stats.endtime
     info['sta'] = data.stats.station
     info['f'] = data.stats.sampling_rate
     info['net'] = data.stats.network
     info['ch'] = data.stats.channel
     info['loc'] = data.stats.location
     info['npts'] = data.stats.npts
   elif mode=='single' and hasattr(data[0], 'stats'):
     tr = data[0]
     info['stt'] = tr.stats.starttime
     info['endt'] = tr.stats.endtime
     info['sta'] = tr.stats.station
     info['f'] = tr.stats.sampling_rate
     info['net'] = tr.stats.network
     info['ch'] = tr.stats.channel
     info['loc'] = tr.stats.location
     info['npts'] = tr.stats.npts
   else:
     info['net'] = get_nets(data) 
     info['f'] =  get_freq(data)
     stt, endt= get_duration(data)
     info['stt']= stt
     info['endt']=endt
     info['sta'], info['loc']= get_stations(data)
     info['ch'] = get_channels(data)
     info['npts'] = int(info['f'] *(info['endt']- info['stt']))  
   return info

def get_snr(tr_vertical, tr_horizontal, pt_p, pt_s, mode='caps',
            snr_pre_window=4, snr_post_window=4, highpass=None):
    """
    This function calculates snr from seismic data.

    tr_vertical: sac trace vertical component

    tr_horizontal: sac trace horizontal component

    pt_p: p phase utcdatetime object

    pt_s: s phase udtdatetime object

    mode: 'caps': calculate similar to CAPSphase; 'std': calculate snr from std; 'sqrt' calculate using sqrt.

    snr_pre_window: the size of window before a phase or noise

    snr_post_windo: the size of window after a phase or signal

    highpass: if data is to be filtered with highpass filter

    returns snr as db if 'caps' mode is used while two values for horizontal and vertical for the other two modes
    """
    if highpass:
        tr_vertical = tr_vertical.filter(
            'highpass', freq=highpass).taper(
                max_percentage=0.1, max_length=0.1)
        tr_horizontal = tr_horizontal.filter(
            'highpass', freq=highpass).taper(
                max_percentage=0.1, max_length=0.1)
    tr_signal_p = tr_vertical.copy().slice( 
        pt_p, pt_p + snr_pre_window )
    tr_signal_s = tr_horizontal.copy().slice( 
        pt_s, pt_s + snr_pre_window ) 
    tr_noise_p = tr_vertical.copy().slice( 
        pt_p - snr_pre_window, pt_p )
    tr_noise_s = tr_horizontal.copy().slice( 
        pt_s-snr_pre_window, pt_s )
  
    if mode.lower() == 'std':
        snr_p = np.std(tr_signal_p.data)/np.std(tr_noise_p.data)
        snr_s = np.std(tr_signal_s.data)/np.std(tr_noise_s.data)

    elif mode.lower() == 'sqrt':
        epsilon=np.finfo('float').eps
        print(np.sum(np.square(tr_noise_p.data)), np.sum(np.square(tr_noise_s.data)))
        if np.sum(np.square(tr_noise_s.data))< 0:
          print(np.square(tr_noise_s.data))
          exit()
        snr_p = np.sqrt(np.square(tr_signal_p.data).sum())\
             / np.sqrt(np.square(tr_noise_p.data).sum()) 
        snr_s = np.sqrt(np.square(tr_signal_s.data).sum())\
             / np.sqrt(np.square(tr_noise_s.data).sum())

    elif mode.lower() == 'caps':
        snr = 10 * np.log10(np.std(tr_signal_s.data)/np.std(tr_noise_p.data))
        return snr
    return snr_p, snr_s

   
def time_format(t):
   '''
   This function returns formatted time string from utc datetime object

   t: utc datetime object

   returns formatted string with time
   '''
   return(f'{t.hour}:{t.minute}:{t.second}:{str(t.microsecond)[:2]}')

def date_format(t):
   '''
   This function returns formatted date string from utc datetime object

   t: utc datetime object

   returns formatted string with date
   '''
   return(f'{t.day}\{t.month}\{t.year}')

def plot_trigger(tr, ptime, stime, file=None):
   """
   This function plots triggers showing P and S arrivals. The plots can be optionally saved to a file.

   tr: obspy Trace object
  
   ptime: arrival of P phase

   stime: arrival of S phase

   file: file name
   """
   tr_len = len(tr.data)
   f = tr.stats.sampling_rate
   ph_trigger = np.zeros((tr_len))
   if int(tr_len - (stime*f)) >  int(ptime *f):
     ph_trigger[int(ptime *f) : int(tr_len - (stime*f))] =1
   else:
     ph_trigger[int(tr_len - (stime*f)):int(ptime *f) ] =1
   if file:
     obspy.signal.trigger.plot_trigger(tr, ph_trigger,0.5, 0.5, outfile=file)
   else:
     obspy.signal.trigger.plot_trigger(tr, ph_trigger,0.5, 0.5)

def read_stream(path=''):
   """
   This function reads stream data from the given path

   path: path to data

   returns obspy stream object
   """
   st = obspy.read(path)
   return st
   
def get_channels(stream):
   """
   This function obtains a list of all channels in the stream

   stream: obspy stream object

   returns a list of channels present in the stream
   """
   channels = []
   for tr in stream:
     if not hasattr(tr, 'stats'):
       for t in tr:
         if t.stats.channel not in channels:
           channels.append(t.stats.channel)
     else:
       if tr.stats.channel not in channels:
           channels.append(tr.stats.channel)  
   return channels

def get_stations(stream):
   """
   This function obtains a list of all stations in the stream

   stream: obspy stream object

   returns a dict with all stations present in the stream and respective available locations
   """
   stations = []
   for tr in stream:
     if not hasattr(tr, 'stats'):
       tr=tr[0]
     if tr.stats.station not in stations:
        stations.append(tr.stats.station)
   loc={}
   for sta in stations:
     loc[sta]=[]
     for tr in stream:
       if not hasattr(tr, 'stats'):
         tr=tr[0]
       if tr.stats.station==sta:
         if tr.stats.location not in loc[sta]:
           loc[sta].append(tr.stats.location)
   return stations, loc

def get_freq(stream):
   """
   This function obtains a list of all sampling rates in the stream

   stream: obspy stream object

   returns a list of sampling rates present in the stream
   """
   freq = []
   for tr in stream:
     if not hasattr(tr, 'stats'):
       tr=tr[0]
     if tr.stats.sampling_rate not in freq:
       freq.append(tr.stats.sampling_rate)
   return freq

def get_duration(stream):
   """
   This function obtains the starting and end times from a trace and the earliest starting and latest end times in case of streams

   stream: obspy stream object

   returns two datatime objects representing starting and end times
   """
   for n, tr in enumerate(stream):
     if not hasattr(tr, 'stats'):
       tr=tr[0]
     if n==0:
       stt = tr.stats.starttime
       endt = tr.stats.endtime
     else:
       if tr.stats.starttime < stt: stt = tr.stats.starttime
       if tr.stats.endtime > endt: endt = tr.stats.endtime
   return stt, endt
    
def get_nets(stream):
   """
   This function obtains a list of all nets in the stream

   stream: obspy stream object

   returns a list of nets present in the stream
   """
   nets = []
   for tr in stream:
     if not hasattr(tr, 'stats'):
       tr=tr[0]
     if tr.stats.network not in nets:       
       nets.append(tr.stats.network)
   return nets

def select_traces(stream, f=[], channels=[], stations=[], check='ps', ptime=0, stime=0):
   """
   This function selects traces present in the stream on the basis of statin, channel and sampling rates

   f: list of sampling rates to select: empty list indicates selecting all rates. 

   channels: list of sampling rates to select: empty list indicates selecting all rates. 
 
   stations: list of statios to select: empty list indicates selecting all.

   check: additional checks to apply: 'ps': rejects samples with length less than p and s times thus indicating corrupted data

   ptime: time for arrival of p phase needed to apply check

   stime: time for arrival of s phase needed to apply check

   stream: obspy stream object

   returns a stream according to the given criterion
   """
   selected_ch = obspy.core.stream.Stream()
   if len(channels)>0:
     for ch in channels:
       selected_ch += stream.select(channel=ch)
   else:
     selected_ch=stream.copy()
   selected_sta = obspy.core.stream.Stream()
   if len(stations)>0:
     for sta in stations:
       selected_sta += selected_ch.select(station=sta)
   else:
     selected_sta = selected_ch
   selected_f = obspy.core.stream.Stream()
   if len(f)>0:
     for fr in f:
       selected_f += selected_sta.select(sampling_rate = fr)
   else:
     selected_f = selected_sta
   
   if check=='ps':
     checked = check_ps(selected_f, ptime=ptime, stime=stime)
     return checked
   else:
     return selected_f

def stream_to_events(st, type='stream', channels=[], remove_duplicates=0, apply_fill=0): 
   """
   This function converts stream to a list with each item corressponding to the same event and station

   events where all required channels are not available are rejected.

   st: stream object

   type: output type; can be 'stream' or 'list'

   channels: channels to select

   remove_duplicates: if duplicates are to be removed

   apply_fill: if events has incomplete data append with zeros

   returns a list of streams
   """  
   # first sorting so channels for same event will end up together
   st.sort(keys=['network', 'station', 'location', 'starttime', 'endtime','channel'])
   
   if len(channels) ==0: 
      channels= get_channels(st)
      
   stream_list=[]
   reject = 0
   over, under = 0, 0
   i=0
   while (i<len(st)-len(channels)):
     print(i)
     print('********************************************************')
     if len(channels)==1:
       stream_list.append(st[i])
       print(f'Trace {i} with channels {channels[0]} is added to list.')
       i+=1         
     else: 
       tr_list=[]
       for n in range(1, len(channels)-1):
         if n==1:
           tr_list.append(st[i])
           print(f'Trace {i} with channel {st[i].stats.channel} collected')
         
         while similarity_check(st[i], st[i+n]):
           print(f'Trace {i} with channel {st[i].stats.channel} is similar to trace {i+n} with {st[i+n].stats.channel}')
           tr_list.append(st[i+n])
           n+=1
           print(i, n, len(st))
           #if (i+n) > len(st)-1: break
            
         if remove_duplicates: tr_list = check_duplicates(tr_list)
         if apply_fill: tr_list = fill(tr_list, channels)  
         if (len(tr_list)==len(channels)):
           print(f'Traces with channels {channels} added to stream')
           
           if type == 'stream': 
             stream = obspy.core.stream.Stream(tr_list)
           elif type == 'list':
             stream=tr_list
           elif type == 'numpy':
             stream=np.asarray(tr_list)
           stream_list.append(stream)
         else:
           print(f'{len(tr_list)} traces rejected as failing the check' )
           if len(tr_list)>len(channels):
             for tr in tr_list: 
               print(tr.stats.channel)
               over+=1 
           else: 
             for tr in tr_list: 
               print(tr.stats.channel)
               under+=1
           #under+=len(tr_list)
           reject+=len(tr_list)   
         i+=n 
   print(f'From {len(st)} traces {len(stream_list) * len(channels)} were used to generate {len(stream_list)} events.')
   print(f'{reject} rejected due to being incomplete. {over} traces has more and {under} has less channels')     
   return stream_list

def get_id(st):
   """
   This function forms a string based on information from a stream or trace

   st: stream or trace

   returns a string based on information from seismic data
   """
   info = get_info(st)
   # trace statistics
   stt = info['stt']
   endt = info['endt']
   sta = info['sta']
   net = info['net']
   loc = info['loc']
   id = f'{stt.year}_{stt.month}_{stt.day}T{stt.hour}_{stt.minute}_{stt.second}_{stt.microsecond}Z.to.{endt.year}_{endt.month}_{endt.day}T{endt.hour}_{endt.minute}_{endt.second}_{endt.microsecond}Z'
   if loc!='':
     id = f'{loc}_{id}'
   if sta != '':
     id = f'{sta}_{id}'
   if net != '':
     id = f'{net}_{id}'
   return id    

def save_stream(st, outpath): 
   """
   This function saves streams as mseed file

   st: obspy stream object

   outpath: path where stream is to be saved
   """
   id = get_id(st)
   filename = os.path.join(outpath, f'{id}.mseed')
   st.write(filename, format='MSEED') 

def check_duplicates(similar_list):
   """
   This function removes duplicate traces from the list of traces

   similar_list: a list with traces

   returns a list of traces
   """
   channels=[]
   new_list =[]
   for tr in similar_list:
     if tr.stats.channel not in channels:
       channels.append(tr.stats.channel) 
       new_list.append(tr)
   return new_list

def fill(similar_list, channels):
   """
   This function copies the data for the last channel if it is not available

   similar_list: list of traces

   channels: required channels

   returns a list with all channels
   """
   new_channels=get_channels(similar_list)
   new_list =[]
   if new_channels == channels:
     return similar_list
   else:
     print(channels, new_channels)
     for ch in channels:
       if ch in new_channels:
         for tr in similar_list:
           if tr.stats.channel==ch: new_list.append(tr)
       else:
         tr = similar_list[-1].copy()
         tr.stats.channel=ch
         new_list.append(tr) 
     
     return new_list

def check_ps(st, ptime=0, stime=0):
   """
   This function checks if trace length is greater than p and stimes

   st: stream object

   ptime: distance of p arrival from trace start

   stime: distance of s arrival from trace end

   returns a stream with checked traces
   """
   checked_st= obspy.core.stream.Stream()
   fail=0
   for tr in st:
     if len(tr)>((ptime+stime)*tr.stats.sampling_rate):
       checked_st+=tr
     else:
       fail+=1
   print(f'From {len(st)} traces {fail} failed and {len(checked_st)} traces passed the check.')
   return checked_st    
   
def similarity_check(tr1, tr2):
   """
   This function checks to see if all stats except channels are similar for two traces

   tr1: traces 1

   tr2: trace 2

   returns 1 if similar and 0 otherwise
   """
   n=0
   for stat in tr1.stats:
     if stat!= 'channel' and stat!= 'mseed':
       if tr1.stats[stat] == tr2.stats[stat]:
         n+=1
         #print(f'{stat} is similar')
       #else:
         #print('{} is not similar'.format(stat))  
   #print('{} are similar from {}'.format(n, len(tr1.stats)))
   if n >= len(tr1.stats)-2:
     return 1
   else:
     return 0

def get_stats(stream_list, ptime=0, stime=0, event_time=60, test_snr=0):
   """
   This function obtains statistics from the given list of streams

   stream_list: given stream of list

   ptime: distance of p arrival from trace start

   stime: distance of s arrival from trace end

   event_time: the limit within which events will be considered part of the same earthquake

   test_snr: if snr is to be tested

   returns a dictionary with the statistics of data
   """
   channels = get_channels(stream_list)
   stream_stats = {}
   stream_stats['len']=[]
   stream_stats['pstime']=[]
   stream_stats['max']=[]
   stream_stats['mean']=[]
   stream_stats['std']=[]
   stream_stats['starts']=[]
   stream_stats['prob_traces']=[]
   stream_stats['events'] = {}
   stream_stats['snr']=[]
   stream_stats['channel'] ={}
   stream_stats['channel']['max']={}
   stream_stats['channel']['mean']={}
   stream_stats['channel']['std']={}
   stream_stats['stations'] ={}
   stream_stats['station'] ={}
   for ch in channels:
     stream_stats['channel']['max'][ch]=[]
     stream_stats['channel']['mean'][ch]=[]
     stream_stats['channel']['std'][ch]=[]
     
   station_info={}
   #events = {}
   lines = []
   stations = [] 
   for n, st in enumerate(stream_list):
     lines = myprint(lines, '****************************************************************************')
     lines = myprint(lines, 'Stream : {}'.format(n))
       
     # get event stats from each stream
     tr = st[0]
     f = tr.stats.sampling_rate
     device = tr.stats.location
     sta = tr.stats.station
     net = tr.stats.network
     stt = tr.stats.starttime
     endt = tr.stats.endtime
     stream_stats['starts'].append(stt)
     if sta not in stations:
       stations.append(sta)
       station_info[sta]={}
       station_info[sta]['num']=[n]
       station_info[sta]['f']=[f]
       station_info[sta]['locations']=[device]
       stream_stats['station'][sta]={}
       stream_stats['station'][sta]['max']={}
       stream_stats['station'][sta]['mean']={}
       stream_stats['station'][sta]['std']={}
       for ch in channels:
         stream_stats['station'][sta]['max'][ch]=[]
         stream_stats['station'][sta]['mean'][ch]=[]
         stream_stats['station'][sta]['std'][ch]=[] 
     else:
       station_info[sta]['num'].append(n)
       if f not in station_info[sta]['f']:
         station_info[sta]['f'].append(f)
       if device not in station_info[sta]['locations']:
         station_info[sta]['locations'].append(device)

     st_length = len(tr.data) 
     stream_stats['len'].append(st_length)  
     st_time = (st_length - (f * (ptime + stime)))/f
     if st_time<0:
       stream_stats['prob_traces'].append(st)
     stream_stats['pstime'].append(st_time)
     data = get_numpy(st)
     st_max = np.max(np.absolute(data))
     st_mean = np.mean(data)
     st_std = np.std(data)
     #for count, ch in enumerate(channels):
     #  if np.max(st[count].data) >st_max: st_max= np.max(st[count].data)
     stream_stats['max'].append(st_max)
     stream_stats['mean'].append(st_mean)
     stream_stats['std'].append(st_std)
     snr = get_snr(st[-1], st[0], stt+ptime, endt-stime)
     stream_stats['snr'].append(snr)
     if test_snr and (snr<0):
       print(snr)
       st.plot()
     lines = myprint(lines, f'The trace is {st_length} sec long ({st_length * f} samples) with frequency {f} Hz')
     lines = myprint(lines, f'The distance between P and S is {st_time} sec long ({st_time*f} samples)')
     lines = myprint(lines, f'The Maximum is {st_max}')
     lines = myprint(lines, f'The snr is {snr}')
     for count, ch in enumerate(channels):
       std = np.std(st[count].data)
       mean = np.mean(st[count].data)
       max_ch = np.amax(np.absolute(st[count].data))
       stream_stats['channel']['std'][ch].append(std)
       stream_stats['channel']['mean'][ch].append(mean)
       stream_stats['channel']['max'][ch].append(max_ch)
       stream_stats['station'][sta]['max'][ch].append(max_ch)
       stream_stats['station'][sta]['mean'][ch].append(mean)
       stream_stats['station'][sta]['std'][ch].append(std)         
       lines = myprint(lines, 'For channel {} Max = {}  Mean = {}  std = {}'.format(ch,
                             stream_stats['channel']['max'][ch][-1], stream_stats['channel']['mean'][ch][-1], stream_stats['channel']['std'][ch][-1]))
   
   data_max = max(stream_stats['max'])
   data_min = min(stream_stats['max'])
   max_d = max(stream_stats['pstime'])
   min_d = min(stream_stats['pstime'])
   max_len = max(stream_stats['len'])
   min_len = min(stream_stats['len'])
   max_snr = max(stream_stats['snr'])
   min_snr = min(stream_stats['snr'])
   lines = myprint(lines, '********************************  Overal Stats ***********************************')
   lines = myprint(lines, 'The overall stats for {} events'.format(n))
   lines = myprint(lines, 'The maximum amplitude for data is {}'.format(data_max))
   lines = myprint(lines, 'The minimum amplitude for data is {}'.format(data_min))
   lines = myprint(lines, 'The maximum distance between phases is {}'.format(max_d))
   lines = myprint(lines, 'The minimum distance between phases is {}'.format(min_d))
   lines = myprint(lines, 'The maximum trace length is {}'.format(max_len))
   lines = myprint(lines, 'The minimum trace length is {}'.format(min_len))
   lines = myprint(lines, 'The maximum snr is {}'.format(max_snr))
   lines = myprint(lines, 'The minimum snr is {}'.format(min_snr))
   # collect events
   e_num = 0
   for i, stt in enumerate(stream_stats['starts']):
     if (i==0):
       stream_stats['events'][str(e_num)]=[i]
       prev_stt = stt
     else:
       if abs(stt - prev_stt) < event_time:
         stream_stats['events'][str(e_num)].append(i)
       else:
         e_num+=1
         stream_stats['events'][str(e_num)]=[i]
         prev_stt =stt
       #event_no.append(i)
   for key in stream_stats['events']:
     if len(stream_stats['events'][key]) > 1:
       print('For event {}, {} streams might belong to the same event'.format(key, len(stream_stats['events'][key])))
       stt_key =[]
       endt_key=[]
       stt_diff = []
       stt_0 = stream_stats['events'][key][0]
       event_st = stream_list[stream_stats['events'][key][0]]
       for count, j in enumerate(stream_stats['events'][key]):
         stt_key.append(stream_stats['starts'][j])
         endt_key.append(stream_list[j][0].stats.endtime)
         stt_diff.append(stream_stats['starts'][j]- stream_stats['starts'][stt_0]) 
         if count >0:
           event_st+=stream_list[int(j)]
       #event_st.plot(automerge=False)
       #print(stt_key, endt_key) 
       #for tr in event_st:
       #  print(tr.stats)       
       
   lines = myprint(lines, '****************************************************************************')
   lines = myprint(lines, 'There are {} problem traces'.format(len(stream_stats['prob_traces'])))
   lines = myprint(lines, 'There are {} events with less than {} sec between start times'.format(key, event_time))
   return stream_stats, lines

def get_unit(name):
    """
    This function returns the unit for different quantities

    name: name of the quantity

    returns a string with unit for the quantity
    """
    unit =''
    if name in ['pstime']: unit = 'sec'
    elif name in ['len']: unit = 'npts'
    elif name in ['snr']: unit = 'dB'
    elif name in ['max', 'min', 'mean', 'std']: unit = ''
    return unit


def plotting_utils():
   """
   ---------------------------------------------- Plotting utils ------------------------------------------------------------

   The following are functions used in making plots
   """

def get_lim(ymin, ymax, offset):
    """
    This function applies an offset to minimum and maximum limits used in making plots

    ymin: minimum value

    ymax: maximum value

    offset: offset added to limits

    returns a list with modified minimum and maximum limits
    """
    return [ymin - (offset*abs(ymax-ymin)), ymax + (offset*abs(ymax-ymin))]

def plot_hist(data, label, unit, bins, plotdir, log=0, scat=0):
    """
    This function plots histogram and saves to the given directory

    data: data to be plotted

    label: data label

    unit: unit for the plotted quantity

    bins: histogram bins

    plotdir: directory where results are saved

    log: if log bins are to be used

    scat: if scatter plots are to be generated
    """
    if log: bins = np.logspace(np.log10(np.min(data)),np.log10(np.max(data)), bins)
    plt.figure()
    if scat:
      plt.scatter(data, data, label = f'{label}:{np.mean(data):.2f} $\pm$ {np.std(data):.2f} for {len(data)} events')
    else:
      plt.hist(data, bins =bins, label = f'{label}:{np.mean(data):.2f} $\pm$ {np.std(data):.2f} for {len(data)} events')
    if unit!= '':
      xlabel= f'{label} [{unit}]'
    else:
      xlabel= label
    plt.xlabel(xlabel)
    if log: plt.xscale("log")
    plt.ylabel('Counts')
    plt.tight_layout() 
    plt.legend()
    plt.savefig(os.path.join(plotdir, remove_spaces(f'{label}.png')))
    plt.close()

def plot_hist_ch(data, axlabels, label, unit, bins, plotdir, log=0, scat=0):
    """
    This function makes channels wise histograms

    data: multi channel data

    axlabels: axis wise labels

    label: label

    unit: unit for the quantity

    bins: histogram bins

    plotdir: directory to save plots

    log: if using log bins

    scat: if making scatter plots
    """
    if log: bins = np.logspace(np.log10(np.min(data)),np.log10(np.max(data)), bins)
    fig, ax= plt.subplots(len(data), 1)
    for i in np.arange(len(data)):
       l=axlabels[i]
       num = len(data[i])
       ax[i].hist(data[i], bins =bins, label = f'{l} for {num} events')
       ax[i].set_ylabel('Counts')
       if log: ax[i].set_xscale("log")
       ax[i].legend()
    if unit!= '':
      xlabel= f'{label} [{unit}]'
    else:
      xlabel= label
    ax[i].set_xlabel(xlabel)
        
    plt.tight_layout() 
    plt.savefig(os.path.join(plotdir, remove_spaces(f'{label}.png')))
    plt.close()

def time_hist(time_list, label, bins, plotdir):
    """
    This functions plots a histogarm of datetime objects

    time_list: list with datetime objects

    label: plot label

    bins: histogram bins
 
    plotdir: directory to save plots
    """
    fig, ax = plt.subplots()
    time_data = mdates.date2num(time_list)
    ax.hist(time_data, bins=bins, label=label)
    ax.set_xticklabels(ax.get_xticks(), rotation=15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
    ax.legend()
    ax.set_ylabel('Counts')
    ax.set_xlabel('date')
    plt.savefig(os.path.join(plotdir, remove_spaces(f'{label}.png')))
    plt.close()

def plot_stats(stats_dict, plotdir, bins=50):
    """
    This function plots the quanitities from the statistics dict

    plotdir: the directory to save plots

    bins: number of bins
    """
    data_list_full =[]
    label_list_full = []
    for key in stats_dict:
      if isinstance(stats_dict[key], list) and key!='prob_traces':
        unit = get_unit(key)
        label= key if key!='starts' else 'EQ events'
        if key=='snr':
          if isinstance(stats_dict[key][0], tuple):            
            stats_dict[key]= np.mean(stats_dict[key], axis=1)
            
        log=1 if key in ['max', 'std'] else 0
        if key in ['starts']:
          time_hist(stats_dict[key], label, bins, plotdir)      
        else:        
          plot_hist(data=stats_dict[key], label=label, unit=unit, bins=bins, plotdir=plotdir, log=log)      
      elif key=='channel':
        for key2 in stats_dict[key]:
          unit = get_unit(key2)
          log=1 if key2 in ['max', 'std'] else 0
          data=[]
          axlabels = []
          for key3 in stats_dict[key][key2]:
            label = f'{key2} {key3}'
            data.append(stats_dict[key][key2][key3])
            axlabels.append(label)
          plot_hist_ch(data, axlabels, key2, unit, bins, plotdir, log=log, scat=0)
      elif key=='station':
        for key2 in stats_dict[key]:
          for key3 in stats_dict[key][key2]: 
            unit = get_unit(key3)
            log=1 if key3 in ['max', 'std'] else 0
            data=[]
            axlabels = []
            for key4 in stats_dict[key][key2][key3]:
              label = f'{key3} {key4}'
              data.append(stats_dict[key][key2][key3][key4])
              axlabels.append(label)              
            plot_hist_ch(data, axlabels, key2+key3, unit, bins, plotdir, log=log, scat=0)  
        
def plot_class_boundries(stream, n, class_fill, plotpath, ylabel=''):
    """
    This function plots the classes defined for seismic trace superimposed on data

    n: stream number

    class_fill: dictionary with class time and color data

    plotpath: directory to save plot

    ylabel: label of y axis
    """
    data = stream.data
    plt.figure(figsize=(9,3))
    alpha = 0.4
    plt.plot(data)
    ymin = min(data)
    ymax = max(data)
    for cl in class_fill:
      plt.axvspan(class_fill[cl]['boundaries'][0], class_fill[cl]['boundaries'][1], alpha=alpha, color=class_fill[cl]['color'], label =f'{cl} class')
      if class_fill[cl]['arrival']>0:
        plt.vlines(class_fill[cl]['arrival'], ymin = 1.2 * ymin, ymax=1.2 * ymax, color=class_fill[cl]['color'], label =f'{cl} arrival')
    plt.xlabel('time [sec * f]')
    plt.ylabel(ylabel)
    plt.ylim(1.2 * ymin, 1.2 * ymax)
    plt.xlim(0, 1.2 * len(data))
    plt.tight_layout() 
    plt.legend()
    plt.savefig(os.path.join(plotpath, f'st_{n}.png'))
    plt.close()

def symlogspace(col, n_cuts, dtype='float64'):
    """
    This function splits a data range into log-like bins but with 0 and negative values
    taken into account. Log cuts start from the closest value to zero.
    
    Parameters

    ----------

    col: df column or array

    n_cuts: int
            Number of cuts to perform

    dtype: dtype of the outputs
    """
    min_val = col.min()
    max_val = col.max()
    
    # compute negative and positive range
    min_pos = col[col > 0].min() if not np.isnan(col[col > 0].min()) else 0
    if min_val < 0:
      max_neg = col[col < 0].max() if not np.isnan(col[col < 0].max()) else 0
    else:
      max_neg = 0
    neg_range = [-min_val, -max_neg] if min_val < max_neg else None
    pos_range = [max(min_val, min_pos), max_val] if max_val > min_pos else None
    
    # If min value is 0 create a bin for it
    zero_cut = [min_val] if max_neg <= min_val < min_pos else []

    n_cuts = n_cuts - len(zero_cut)

    neg_range_size = (neg_range[0] - neg_range[1]) if neg_range is not None else 0
    pos_range_size = (pos_range[1] - pos_range[0]) if pos_range is not None else 0
    range_size = neg_range_size + pos_range_size

    n_pos_cuts = max(2, int(round(n_cuts * (pos_range_size / range_size)))) if range_size > 0 and pos_range_size > 0 else 0
    # Ensure each range has at least 2 edges if it's not empty
    n_neg_cuts = max(2, n_cuts - n_pos_cuts) if neg_range_size > 0 else 0   
    # In case n_pos_cuts + n_neg_cuts > n_cuts this is needed
    n_pos_cuts = max(2, n_cuts - n_neg_cuts) if pos_range_size > 0 else 0

    neg_cuts = []
    #print(neg_range, n_neg_cuts)
    if n_neg_cuts > 0:
        neg_cuts = list(-np.logspace(np.log10(neg_range[0]), np.log10(neg_range[1]), n_neg_cuts, dtype=dtype))

    pos_cuts = []
    if n_pos_cuts > 0:
        pos_cuts = list(np.logspace(np.log10(pos_range[0]), np.log10(pos_range[1]), n_pos_cuts, dtype=dtype))

    result = neg_cuts + zero_cut + pos_cuts
   
    # Add 0.1% to the edges to include min and max
    result[0] = min(result[0] * 1.1, result[0] * 0.9)
    result[-1] = result[-1] * 1.1
    return result

def training_utils():
   """
   ---------------------------------------------- training utils --------------------------------------------------------

   The following are functions help in training
   """

def divide_data(x, y, split_ratio):
   """
   This function slices input and output data according to split ratio

   x: inputs

   y: outputs

   split_ratio: a list with ratio for train, validation and test segments

   returns split data
   """
   test_val =  1-split_ratio[0]    
   x_train, x_test, y_train, y_test = train_test_split(x, y,
   test_size=test_val, shuffle = True, random_state = 8)
     
   test_val = split_ratio[2]/(split_ratio[2]+split_ratio[1])
   # Use the same function above for the validation set
   x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, 
   test_size=test_val, random_state= 8)
   #data = [x_train, x_val, x_test, y_train, y_val, y_test]
   return x_train, x_val, x_test, y_train, y_val, y_test


def MAPE(y_true, y_pred, multioutput=""):
    """
    "MAPE Definition"
    """
    y_true, y_pred = np.squeeze(y_true), np.squeeze(y_pred)
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, weights=None, axis=0)
    
    error = np.average(output_errors)
   
    mape = np.average(error)*100
    
    return mape

################################### main ##########################################################

def main():
   """
   Here the functionality of different functions can be testes
   """
if __name__ == '__main__':
  main()