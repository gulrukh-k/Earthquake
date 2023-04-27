"""
A class to read seed data from a given path and obtain basic specs.
"""
import numpy as np
import obspy
from obspy.core.stream import Stream
import os
import common as com
from obspy.signal.trigger import pk_baer, plot_trigger
import matplotlib.pyplot as plt

# Traces can be selected on the basis of station, channel and frequency.
# A list of streams can be obtained where each stream corressponds to different channels of the same signal

class SeedDataSet:
   def __init__(self, path='', ptime=60, stime=300):
     """
     path: path to data

     ptime: time in sec from the start to P arrival

     stime: time in sec from the end to S arrival
     """
     self.path = path
     if self.path != '':
       self.stream = self.read_stream()
       
       self.channels = self.get_channels()
       self.stations = self.get_stations()
       self.freq = self.get_freq()
       self.len = len(self.stream)
       self.ptime = ptime
       self.stime = stime
       self.name = self.path
   
   def read_stream(self):
     """
     This function reads the data from the dataset path into stream
     """
     st = com.read_stream(self.path)
     return st
   
   def get_channels(self):
     """
     This function obtains a list of all channels in the data
     """
     channels = com.get_channels(self.stream)
     return channels

   def get_stations(self):
     """
     This function obtains a list of all stations and their locations
     """
     stations, loc = com.get_stations(self.stream)
     return stations, loc

   def get_freq(self):
     """
     This function obtains a list of all sampling rates in the data
     """
     freq = com.get_freq(self.stream)
     return freq

   # select traces on the basis of statin, channel and sampling rates
   def select(self, f=[], channels=[], stations=[], check='ps'):
     """
     This function selects traces on the basis of statin, channel and sampling rates
  
     f: list of sampling rates to select (empty list selects all)

     channels: list of channels to select (empty list selects all)

     stations: list of stations to select (empty list selects all)

     check: additional check; 'ps': implements the check; trace length should be greater than ptime + stime otherwise data is corrupted

     returns obspy stream object with selected traces
     """
     selected = com.select_traces(self.stream, f=f, channels=channels, stations=stations, 
                                  check=check, ptime=self.ptime, stime=self.stime)
     
     return selected

   def stream_to_list(self, f=[], channels=[], stations=[], type='stream', check='ps', apply_fill=1, remove_duplicates=1, norm=None):
     """
     This function convert stream to a list with each item corressponding to the same event and station. events where all required channels are not available are rejected.

     f: list of sampling rates to select (empty list selects all)

     channels: list of channels to select (empty list selects all)

     stations: list of stations to select (empty list selects all)

     type: type of data to return; currently only 'stream' is available

     check: additional check; 'ps': implements the check; trace length should be greater than ptime + stime otherwise data is corrupted

     apply_fill: 0: events with less channels than required are rejected; 1: the last channel is copied in missing channels

     remove_duplicates: 0: do not check for duplicate traces; 1: removes events with duplicate traces

     norm: None: no normalization; 'local': normalize the stream 

     returns a list with obspy stream objects with selected traces for same event
     """
     st = self.select(f=f, channels=channels, stations=stations, check=check) 
     if norm=='global':
       st=st.normalize(global_max=True)
     st_list = com.stream_to_events(st, channels=channels, apply_fill=apply_fill, remove_duplicates=remove_duplicates)
     print(f'{len(st)} traces were selected from {self.len} traces available in dataset ')
     if norm=='local':
       for st in st_list: st.normalize()      
     return st_list 

   def get_stats(self, outdir=None, f=[], channels=[], stations=[]):
     """
     This functions obtains statistics of the dataset and saves the results in outdir

     outdir: result directory

     returns a ditionary with stats and a text log
     """
     stream_list = self.stream_to_list(f=f, channels=channels, stations=stations)
     stats, log = com.get_stats(stream_list, ptime=self.ptime, stime=self.stime)
     if outdir: 
       com.plot_stats(stats, outdir)
       com.text_to_file(log, os.path.join(outdir, 'info.txt'))
     return stats, log

def main():
   """
   Small working example:

   # define path for seed data

   dataPath = "..\\SeedData\\PeshawarData2016_2019\\*\\*.mseed" 

   # directory to save results

   outdir = os.path.join('plots', 'PeshawarData2016_2019')

   com.safe_mkdir(outdir)

   # Dataset object with the path and phase arrivals

   dataset = SeedDataSet(dataPath, ptime=60, stime=200) 
  
   # get the statistics

   stats, log = dataset.get_stats(outdir)   
   """
   dataPath = "..\\SeedData\\Jan2018-Jan2020_seed\\*\\*.mseed"
   #dataPath = "..\\SeedData\\Jan2018-Jan2020_seed\\mseed4testing\\*.mseed"
   #dataPath = "..\\SeedData\\PeshawarData2016_2019\\*\\*.mseed"
   dataPath = "..\\SeedData\\ISBData2018_2021\\*\\*.mseed" 
   #channels = ['BH1', 'BH2', 'BHZ']
   #channels = []
   channels = ['HHE', 'HHN', 'HHZ']
   stations =[]
   f = []
   outdir = os.path.join('dataset_info', 'ISB_data_all')
   com.safe_mkdir(outdir)
   dataset = SeedDataSet(dataPath, ptime=60, stime=120)   
   stats, log = dataset.get_stats(outdir, channels=channels, stations=stations, f=f)   

if __name__ == '__main__':
   main() 