import numpy as np
import os
import csv

num_channels = 14
sample_duration = 1
sampling_rate = 128

def dump_binaries():
    """
    Converts the exported CSV files from the Emotiv app into numpy binary files for faster loading
    """
    file_location = "C:/Users/camwu/Downloads/Temp/EmotivOutput"
    result = {i: [] for i in range(1, 10)}
    for filename in sorted(os.listdir(file_location)):
        if filename.endswith(".csv"):
            _class = int(filename.split("-")[1][1])
            all_rows = []
            with open(os.path.join(file_location, filename)) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for i, row in enumerate(reader):
                    if i < 2: continue
                    all_rows.append(row[4:18])
            
            # First class had slightly different cutoff (first time using the app)
            if _class == 1:
                if len(result[1]) == 0:
                    all_rows = all_rows[256:(128*60*5)-256]
                else:
                    all_rows = all_rows[256:(256*3)+(128*60*5)]
            else:
                all_rows = all_rows[256:256+(128*60*5)]
            result[_class].extend(all_rows)
    
    for key in range(1, 10):
        np.save(f"eeg_data_{key}.npy", np.array(result[key], dtype=np.float64))

def load_real_eeg_data():
    """
    Loads the eeg data from the binaries into numpy arrays
    """
    eeg_data = []
    labels = []
    for key in range(1, 10):
        data = np.load(f"real/binaries/eeg_data_{key}.npy")
        # split data into 128 row intervals
        data = [data[i:i+128] for i in range(0, len(data), 128)]
        eeg_data.extend(data)
        labels.extend([key-1 for _ in range(len(data))])
    return np.array(eeg_data), np.array(labels)
