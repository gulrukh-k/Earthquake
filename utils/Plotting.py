"""
This class plots the results of the test bench
"""
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import common as com

class Plotting():
   def __init__(self, test, size=None, trigger ='sta/lta', 
                phase_seq = ['p', 's', 'n', 't'], phase_colors = ['r','b','green', 'pink']):
     """
     test: Testbench object

     size : size of the plot

     trigger : type of the trigger to plot in probe only 'sta/lta' implemented

     phase_seq: the phase order for the plots

     phase_colors: the corrssponding colors for phase sequence
     """
     self.test = test
     self.size = size
     self.trigger = trigger
     self.phase_seq= phase_seq
     self.phase_colors= phase_colors
    
   def no_pick(self, leg_text, label, color):
     """
     This function generats text to display for a model without any picks

     leg_text: previous text for legend
   
     label: label for pick

     color: color for pick

     returned appended legend text
     """
     line = Line2D([0], [0], label=label + ' none', color=color)
     leg_text.append(line)
     return leg_text

   def add_markers(self, axis, picks, ymin, ymax, color, label, leg_text):
     """
     This function adds picks to the plot as vertical markers.

     axis: current matplotlib axis object

     picks: list with picks

     ymin, ymax : limits for the markers

     color: marker color

     label: label for the pick

     leg_text: in case of no picks just text is added to legend

     returns updated legend text
     """
     if (len(picks) >0): # if pick is not empty
       if picks[0]!=0: # if pick is not equal to 0
         axis.vlines(picks, ymin = [ymin]* len(picks), ymax=[ymax]* len(picks), 
                                   color=color, label =label)
       else:
         leg_text= self.no_pick(leg_text, label=label, color=color)
     else:           
       leg_text= self.no_pick(leg_text, label=label, color=color)       
     return leg_text

   def add_markers_dash(self, axis, picks, ymin, ymax, color, label, leg_text):
     '''
     This function adds markers to the plot with flat dashes at the end points useful when picks overlap.
     
     axis: current axis

     picks: list with picks

     ymin, ymax : limits for the markers

     color: marker color

     label: label for the pick

     leg_text: in case of no picks just text is added to legend

     returns updated legend text
     '''
     x=[]
     y = []
     for pick in picks:
       x.append(pick)
       x.append(pick)
       y.append(ymin)
       y.append(ymax)
     if (len(picks) >0): # if pick is not empty
       if picks[0]!=0: # if pick is not equal to 0
         axis.vlines(picks, ymin = [ymin]* len(picks), ymax=[ymax]* len(picks), 
                                   color=color, label =label)
         axis.scatter(x, y, marker ='_', color=color)
       else:
         leg_text= self.no_pick(leg_text, label=label, color=color)         
     else:           
       leg_text= self.no_pick(leg_text, label=label, color=color) 
     return leg_text

   def add_probe(self, key, ax):
     """
     This function plots the output values representing the probability of a phase

     key : mode or type

     ax: matplotlib axis object
     """
     model_win = self.test.models_dict[key]['model'].win # window size of selected model         
     for count, phase in enumerate(self.phase_seq): # follow the phase sequence
       if phase in self.test.models_dict[key]['model'].phases: # if phase exits for the model
         if len(self.test.picks_dict[key][phase]['out']) > 0: # if there is any output available for the phase
           if model_win > self.test.interval:   # if model window is greater than interval then output needs to be sorted in chronological order
             ind = np.argsort(np.array(self.test.picks_dict[key][phase]['out_loc']))
             out = np.array(self.test.picks_dict[key][phase]['out'])[ind]
             out_t = np.array(self.test.picks_dict[key][phase]['out_loc'])[ind]
           else:
             out = self.test.picks_dict[key][phase]['out'] 
             out_t = self.test.picks_dict[key][phase]['out_loc']           
           
           ax.plot( out_t, out, linewidth=1, label =f'{phase} {key}', color=self.phase_colors[count])
           if phase in self.test.models_dict[key]['model'].peak_thresh:  # also plot thresholds for the phases
             ax.axhline(y = self.test.models_dict[key]['model'].peak_thresh[phase], color = self.phase_colors[count], linestyle = 'dashed')  
     if len(self.test.triggers[self.trigger][key])>0: # if trigger was applied then plot trigger values
       ax.set_ylim(-0.1, 1.5)
       #print(len(out_t), self.test.triggers[self.trigger][key])
       ax.plot(out_t, self.test.triggers[self.trigger][key], linewidth=1, label =f'{self.trigger} {key}', color='maroon', linestyle = 'dashed')
     else:
       ax.set_ylim(-0.1, 1.0)

   def add_fill(self, axis, picks, ymin, ymax, color, label, leg_text, fill_width):
     '''
     This function add picks to plot as filled regions

     axis: matplotlib axis

     picks: list with picks

     ymin, ymax : limits for the markers

     color: marker color

     labe: axis label

     empty: in case of no picks just text is added to legend

     returns updated legend text
     '''
     if (len(picks) >0) or (' n' in label):
       axis.vlines(picks, ymin = [ymin]* len(picks), ymax=[ymax]* len(picks), 
                                   color=color, label =label)
       for ind, pick in enumerate(picks):
         axis.axvspan(pick-(fill_width/2), pick+(fill_width/2), alpha=0.5, color=color)
     else:           
       leg_text= self.no_pick(leg_text, label=label, color=color)
     return leg_text
   
   def col_set1(self, x):
     """
     This function obtains a colors based on a pallette

     x: length of data 

     returns colors
     """
     return plt.cm.Set1(np.linspace(0,1, 2 *x)) 


   def multi_markers_dash(self, ax, key_list, ymin, ymax, leg_text):
     '''
     This function plots markers from multiple models on the same plot

     ax: axis

     key_list: list of model keys to plot

     ymin, ymax: data min and max

     leg_text : legend text

     returns updated legend text and an empty list
     '''
     colors = self.col_set1(len(key_list)) 
     offset, col= 0, 0
     for key in key_list:    
       ylim = com.get_lim(ymin, ymax, offset)          
       leg_text = self.add_markers_dash(ax, self.test.picks_dict[key]['p']['pick'], ylim[0], ylim[1], colors[col], key + ' P', leg_text)
       if col==0:
         leg_text = self.add_markers_dash(ax, self.test.picks_dict[key]['s']['pick'], ylim[0], ylim[1], 'b', key + ' S', leg_text)
       else:
         leg_text = self.add_markers_dash(ax, self.test.picks_dict[key]['s']['pick'], ylim[0], ylim[1], colors[col+1], key + ' S', leg_text)
       col+=2
       offset+=0.05
     key_list=[] # empty the list after plotting
     return leg_text, key_list 
     
   def plot_picks(self, stream, plotpath, num=0, probe = [], snr_list=[]):
     '''
     This is the function for the generation of all plots

     stream: stream in seed format

     plotpath: path to save plots

     num: stream id

     probe: modes that should be probed in detail
     '''     
     keys = list(self.test.picks_dict.keys()) # all keys for which picks are available
     if hasattr(stream, 'stats'):
       stream= [stream]
     probe = [p for p in probe if p in self.test.picks_dict]
     # sorting in different types of modes in order to update the plot format
     obs_keys =[] # multiple modes can be plotted on same panel
     arru_keys = [] # multiple modes can be plotted on same panel
     class_keys = [] # multiple modes cannot be plotted on same panel
     rows = 0 # the number of rows required
     for key in keys:
        if key in ['truth', 'pkbaer', 'arpick', 'stalta_slice', 'pkbaer_slice', 'arpick_slice']:
          obs_keys.append(key) # to add as markers to first plot
        elif ('arru' in key) or ('stalta' in key):
          arru_keys.append(key) # to add as markers to second plot
        else:
          class_keys.append(key) # to add as filled region to one plot per classifier
     
     # calculating additional panels required
     if len(obs_keys)>0:
       rows+=1
     if len(arru_keys)>0:
       rows+=1
     for i in range(len(class_keys)):
       rows+=1
     if rows<len(stream): rows =len(stream)
     rows+=len(probe)

     if not self.size:
       self.size= (12, rows*1.5)
     st_data = []
     st_label = []
     '''
     if hasattr(stream, 'stats'):
       st_data.append(stream.data)
       st_label.append(stream.stats.channel)
     else:
     '''
     for i in range(len(stream)):
         st_data.append(stream[i].data)
         st_label.append(stream[i].stats.channel)
     st_label = st_label + ((rows-i)*[stream[i].stats.channel]) + probe
     
     # get stream info
     info=com.get_info(stream[0])
     fig, ax = plt.subplots(rows, 1, figsize=self.size, sharex=True)
     
     if 'truth' in self.test.picks_dict:    # if true picks are known then snr can be calculated
       snr = com.get_snr(stream[-1], stream[0], 
                      self.test.picks_dict['truth']['p']['utc'][0], self.test.picks_dict['truth']['s']['utc'][0])
     else:
       snr =0
     key_count = 0
     title = com.make_title_info(info, snr)
     snr_list.append(snr)
     for i in range(rows):
       if i < rows-len(probe): 
         if i < len(st_data):
           ax[i].plot(st_data[i], linewidth=1)
           ymax=max(st_data[i])
           ymin = min(st_data[i])
         else:
           ax[i].plot(st_data[-1], linewidth=1)         
         
         leg_text = []       
         if i ==0:           
           ax[i].set_title(title)
         # if obspy picks are available then plot first
         if len(obs_keys)>0:
           leg_text, obs_keys = self.multi_markers_dash(ax[i], obs_keys, ymin, ymax, leg_text) # after plotting the list will be returned empty

         # look for arru picks next         
         elif len(arru_keys)>0:
           leg_text, arru_keys = self.multi_markers_dash(ax[i], arru_keys, ymin, ymax, leg_text)
        
         # finally plot classifier picks     
         elif (len(class_keys)> 0) and (key_count<len(class_keys)) :             
           key = class_keys[key_count]
           if (len(st_data[0]) > (self.test.models_dict[key]['model'].win + self.test.interval)* info['f']):
             fill_width = self.test.interval * info['f']
           else:
             fill_width = self.test.models_dict[key]['model'].win * info['f']
           for count, phase in enumerate(self.test.picks_dict[key]): 
             leg_text = self.add_fill(ax[i], self.test.picks_dict[key][phase]['pick'], ymin, ymax, self.phase_colors[count], f'{key} {phase}', leg_text, fill_width) 
           key_count +=1  # so that next model is plotted in next iteration
       else:
         key = probe[i-(rows-len(probe))]             
         if key in self.test.picks_dict: # plot probes
           self.add_probe(key, ax[i])
           
       handles, labels = ax[i].get_legend_handles_labels()
       handles.extend(leg_text) # extend the legend if empty picks
       ax[i].set_ylabel(st_label[i])
       #if i != len(ax)-1: ax[i].set_xticks([])
       ax[i].legend(handles=handles, loc='upper right')
       ax[i].set_xlabel('npts')
     plt.tight_layout()
      
     plt.savefig(os.path.join(plotpath, 'Picks_stream{}.png'.format(num)))  
     plt.close() 
     

def main():
   '''
   Some code used during development for testing purpose
   '''
   dataPath = "..\\SeedData\\Jan2018-Jan2020_seed\\mseed4testing\\*.mseed"
   stations= ['AKA', 'NIL', 'SIMI']
   channels = ['BH1', 'BH2', 'BHZ']
   f = 40
   dataset = SeedDataSet(dataPath)
   stream_list = dataset.stream_to_list(f=f, channels=channels, stations=stations)
   info = dataset.info
   picker_weights = 'models\\arru\\weights\\train_pick.hdf5'
   detector_weights = 'models\\arru\\weights\\train_multi.hdf5'
   pick_mode = ['arrupick']
   model_dict={}
   for mode in pick_mode:
      if 'arru' in mode:
        arru_model = AppARRU(mode, single=0, pthresh=0.8, sthresh = 0.7)
        if mode == 'arrupick':
          model_dict = arru_model.get_model(model_dict, picker_weights)
        else:
          model_dict = arru_model.get_model(model_dict, detector_weights)
   lines=[]
   for n, st in enumerate(stream_list[::]):
     lines = com.myprint(lines, '****************************************************************************')
     lines = com.myprint(lines,'Stream : {} '.format(n))
     for mode in model_dict:     
       picks = arru_model.get_picks(st, model_dict, dataset)
     
     
if __name__ == '__main__':
   main() 