import obspy
import common as com
import os

st = obspy.read('data\\test_traces\\*.mseed')
new_dir = 'data\\new_test_data'
com.safe_mkdir(new_dir)
for tr in st[::]:
  info = com.get_info(tr)
  name = com.make_trace_info(info)+ '.mseed'
  tr.write(os.path.join(new_dir, name), format='MSEED')
  