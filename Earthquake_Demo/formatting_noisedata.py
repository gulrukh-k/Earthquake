import obspy
import common as com
import os

st = obspy.read('data\\noise_data\\*.msd')
new_dir = 'data\\new_noise_data'
length = 300
channel='HHZ'
st = st.select(channel=channel)
com.safe_mkdir(new_dir)
for tr in st[::]:
  info = com.get_info(tr)
  start = info['stt']
  end = info['endt'] 
  
  while start < end-length: 
    slice = tr.slice(start, start + length)   
    info = com.get_info(slice)
    name = com.make_trace_info(info)+ '.mseed'
    slice.write(os.path.join(new_dir, name), format='MSEED')
    start = start+length