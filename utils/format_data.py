import common as com

old_dir = '..\\SeedData\\PeshawarData2016_2019\\NewTesting_2019\\*.mseed'
new_dir = '2019_test_traces'
com.save_seed_events(old_dir, new_dir, channels=['HHN', 'HHE', 'HHZ'], num=30)