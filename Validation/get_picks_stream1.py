"""
This is a script to validate different methodologies for phase picking on continous seismic traces.

The script uses the SeedDataset, Testbench and Plotting classes from utils folder.

The common.py from utils is a collection of common fuctions.
"""
import numpy as np
import sys
import os
sys.path.append('..\\utils\\')
from SeedDataset import SeedDataSet
from Plotting import Plotting
from Testbench import Testbench
import common as com

def main():
   """
   This code validates selected methodologies on continous traces consisting of following steps:

   - define a SeedDataSet object for the selected dataset

   - obtain a list of streams with selected channels, frequency and channels

   - select the algorithms using modes

   - build the required directory structure

   - define a Testbench object with selected modes

   - define a Plotting object

   - for each stream in list:

          - get the picks obtained by each algorithm

          - plot the picks on actual stream

   - evaluate picks for each version

   - write the evaluation results as well as test configuration to text files

   In order to use this code following steps needs to be taken:

   - select the dataset using dataPath

   - channel, stations and frequencies can be selected. Empty list would indicate to select all available traces.

   - select the modes using identifiers. The test bench and algirthm classes needs to be added if a new approach is to be tested. Currently following modes are available:
          - truth : true arrivals if available. If not available then evaluation will not be done but the picks can be plotted.
  
          - pkbaer : picker for P arrival available from https://docs.obspy.org/packages/autogen/obspy.signal.trigger.pk_baer.html 

          - arpick : picker for P and S arrival available from https://docs.obspy.org/packages/autogen/obspy.signal.trigger.ar_pick.html

          - arrupick: picker for P and S arrival available from https://github.com/tso1257771/Attention-Recurrent-Residual-U-Net-for-earthquake-detection/tree/main/pretrained_model/paper_model_ARRU_20s

          - arrudetect: picker for P and S arrival available from https://github.com/tso1257771/Attention-Recurrent-Residual-U-Net-for-earthquake-detection/tree/main/pretrained_model/multitask_ARRU_20s

          - stftcnn1 - stftcnn6 : different versions of cnn models trained on stft of seismic traces

   - name the plotdir as result file
   """
   # define data path here   
   iristestPath = "..\\SeedData\\Jan2018-Jan2020_seed\\mseed4testing\\*.mseed"  #IRIS test
   iristrainPath = "..\\SeedData\\Jan2018-Jan2020_seed\\mseedFiles\\*.mseed"     #IRIS train
   peshalldataPath = "..\\SeedData\\PeshawarData2016_2019\\*\\*.mseed"             #PESH all
   paktestPath = "..\\SeedData\\paktestData\\*.mseed"             #PESH test
   peshtest1Path = "..\\CSV_datasets\\train_dataset\\Peshawar_dataset_train_100hz_0filt_4s_HHZchannels_half_tails_postnorm_max\\test_traces\\*.mseed" # PESH test
   seedtestPath = "..\\CSV_datasets\\SeedTestTraces\\*.mseed"
   peshtest2Path = "..\\CSV_datasets\\train_dataset\\pesh\\Peshawar_dataset_shuffled_train_100hz_0filt_4s_HHZchannels_half_tails_postnorm_max\\test_traces\\*.mseed"
   peshtest3Path = "..\\CSV_datasets\\train_dataset\\pesh\\PESH_train_100hz_0filt_4s_HHallchannels_factor2_nperf1_tails_postnorm_max\\test_traces\\*.mseed"
   isbtestPath = "..\\CSV_datasets\\train_dataset\\isb\\ISB_train_100hz__0.1_1filt_4s_HHZchannels_factor10_nperf2_tails_postnorm_max\\test_traces\\*.mseed"
   dataPath =peshtest2Path
   plotdir = 'PESH_caps_test2' # result folder
   ptime=60
   if 'ISB' in plotdir: stime = 120
   elif 'PESH' in plotdir: stime = 200
   elif 'IRIS' in plotdir: stime = 300
   stations= []#'AKA', 'NIL', 'SIMI'] # stations selected
   channels = []#'BH1', 'BH2', 'BHZ'] # channels selected
   f = []#40] # frequencies selected
   overlap =4 # interval between algorithm application

   dataset = SeedDataSet(dataPath, ptime=ptime, stime=stime)   # dataset object
   
   stream_list = dataset.stream_to_list(f=f, channels=channels, stations=stations) # obtain stream list 
   
   stream_list= stream_list[::] # a certain number of traces can be selected
   
   pick_modes = ['truth', 'pkbaer', 'stalta_slice', 'caps'] # modes selected 
   probe_list = ['caps'] # modes whose output is plotted directly
   plotpath = os.path.join('results', f'{plotdir}_overlap{overlap}') #
   com.safe_mkdir(plotpath)     
   
   test = Testbench(pick_modes=pick_modes, overlap=overlap) # Testbench object
   plots = Plotting(test) # Plotting results
   log=[]
   
   # get picks from each stream and plot results
   for n, st in enumerate(stream_list):     
     lines = com.myprint(log, '***************************************************************************************')
     lines = com.myprint(log,'Stream : {} '.format(n))
     
     info = com.get_info(st)      
     
     log = test.get_picks(st, ptime=info['stt'] + dataset.ptime, stime=info['endt'] - dataset.stime, log=log)
     
     #test.display_picks() # can be used to simply display picks
     plots.plot_picks(st, plotpath, n, probe=probe_list)
   
   # evaluate picks and write to file
   log = test.evaluate(log)
   com.to_file(log, os.path.join(plotpath, 'evaluation.txt')) 

   # write class configuration to a text list 
   data_config = com.get_class_config(dataset, [com.decorate('Data Set Configuration')])  
   test_config = com.get_class_config(test, [com.decorate('Test Configuration')]) 
   config = data_config + test_config 
   arch = []
   for mode in test.models_dict:
      config = config + com.get_class_config(test.models_dict[mode]['model'], [com.decorate(f'{mode} Configuration')])
      if hasattr(test.models_dict[mode]['model'], 'model'):
        arch.append(com.decorate(f'{mode} Architecture')) 
        test.models_dict[mode]['model'].model.summary(print_fn=lambda x: arch.append(x + '\n'))
        short_model_summary = "\n".join(arch)
   
   # write configurations to text file
   com.to_file(arch, os.path.join(plotpath, 'architectures.txt'))
   plot_config =  com.get_class_config(plots, [com.decorate('Plotting Configuration')]) 
   config = config + plot_config 
   com.to_file(config, os.path.join(plotpath, 'config.txt'))
  
if __name__ == '__main__':
   main() 