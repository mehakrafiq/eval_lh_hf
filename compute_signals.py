import numpy as np
import pandas as pd
import ast as ast
from scipy.signal import hilbert, chirp
from scipy.signal import butter, lfilter
from scipy.interpolate import CubicSpline


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def compute_hilbert_imag(signal):
    analytic_signal = hilbert(signal)
    return analytic_signal.imag
    

def remove_outliers(window, percent):
    top = int(percent*len(window))
    vals_min =  np.argpartition(window,-top)[-top:]
    vals_max = np.argpartition(window,top)[:top]
    new = np.delete(window,[vals_min,vals_max])
    return new

def compute_mean(window):
    return np.mean(window)


def create_new_signal(dataframe,user, sections):
    data_sub = dataframe[dataframe.user==user]
    arr = np.array([])
    new = []
    for i in sections:
        arr = np.concatenate((arr,data_sub[data_sub.section==i].rr_array))
    for idx in range(len(sections)):
        new = new + list(arr[idx])
    return np.array(new)

def create_dict_user(dataframe):
    users = dataframe.user.unique()
    x = {'x':None}
    y = {'y':None}
    user_dict = dict.fromkeys(users,{'x':None,'y':None})
    return user_dict

def construct_signal_all(dataframe, sections):
    arr = []
    users =  dataframe.user.unique()
    for i in users:
        vals = create_new_signal(dataframe, i, sections)
        arr.append(vals)
    return arr
        


def compute_lf(data,lf):
    vals = []
    for i,j in enumerate(data.rr_array):
        if(data.fs[i]>=1):
            bp =  butter_bandpass_filter(j,lf[0],lf[1],data.fs[i])
            vals.append(bp)
        else:
            vals.append('N/A')
    data['bandpass_lf'] =  vals
    return

def compute_lf_array(data,lf,signal):
    vals = []
    for i,j in enumerate(data[signal]):
        if(data.fs[i]>=1):
            bp =  butter_bandpass_filter(j,lf[0],lf[1],data.fs[i])
            vals.append(bp)
        else:
            vals.append('N/A')
        
    data['bandpass_lf_'+signal] =  vals
    return

def compute_hf(data, hf):
    vals = []
    for i,j in enumerate(data.rr_array):
        if(data.fs[i]>=1):
            bp =  butter_bandpass_filter(j,hf[0],hf[1],data.fs[i])
            vals.append(bp)
        else:
            vals.append('N/A')
    data['bandpass_hf'] =  vals
    return

def compute_hf_array(data,hf,signal):
    vals = []
    for i,j in enumerate(data[signal]):
        if(data.fs[i]>=1):
            bp =  butter_bandpass_filter(j,hf[0],hf[1],data.fs[i])
            vals.append(bp)
        else:
            vals.append('N/A')
    data['bandpass_hf_'+signal] =  vals
    return

def compute_hilbert_lf_ia(data):
    vals = []
    for i,j in enumerate(data.bandpass_lf):
        if(data.fs[i]>=1):
            analytic_signal = hilbert(j)
            vals.append(np.abs(analytic_signal))
        else:
            vals.append('N/A')
    data['LF_IA'] = vals
    return

def compute_hilbert_lf_ia_array(data,signal):
    vals = []
    for i,j in enumerate(data[signal]):
        if(data.fs[i]>=1):
            analytic_signal = hilbert(j)
            vals.append(np.abs(analytic_signal))
        else:
            vals.append('N/A')
    data['LF_IA_'+signal] = vals
    return
        
def compute_hilbert_hf_ia(data):
    vals = []
    for i,j in enumerate(data.bandpass_hf):
        if(data.fs[i]>=1):
            analytic_signal = hilbert(j)
            vals.append(np.abs(analytic_signal))
        else:
            vals.append('N/A')
    data['HF_IA'] = vals
    return

def compute_hilbert_hf_ia_array(data,signal):
    vals = []
    for i,j in enumerate(data[signal]):
        if(data.fs[i]>=1):
            analytic_signal = hilbert(j)
            vals.append(np.abs(analytic_signal))
        else:
            vals.append('N/A')
    data['HF_IA_'+signal] = vals
    return
        

def compute_window_lf_ia(data, window_size):
    vals = []
    for i,j in enumerate(data.LF_IA):
        if(data.fs[i]>=1):
            arr =  sliding_windows(j, window_size)
            vals.append(arr)
        else:
            vals.append('N/A')
    data['LF_window'] = vals
    return 

def compute_window_lf_ia_array(data, signal, window_size):
    vals = []
    for i,j in enumerate(data[signal]):
        if(data.fs[i]>=1):
            arr =  sliding_windows(j, window_size)
            vals.append(arr)
        else:
            vals.append('N/A')
    data[signal+str(window_size)] = vals
    return 
    
            
        
def compute_window_hf_ia(data, window_size):
    vals = []
    for i,j in enumerate(data.HF_IA):
        if(data.fs[i]>=1):
            arr =  sliding_windows(j, window_size)
            vals.append(arr)
        else:
            vals.append('N/A')
    data['HF_window'] = vals
    return 

def compute_window_hf_ia_array(data, signal, window_size):
    vals = []
    for i,j in enumerate(data[signal]):
        if(data.fs[i]>=1):
            arr =  sliding_windows(j, window_size)
            vals.append(arr)
        else:
            vals.append('N/A')
    data[signal+str(window_size)] = vals
    return 
    
    
def compute_lf_windows(data, window_size):
    vals = []
    for i,j in enumerate(data.LF_IA):
        if(data.fs[i]>=1):
            arr =  sliding_windows(j, window_size)
            vals.append(arr)
        else:
            vals.append('N/A')
    return vals

def compute_feature_windows(data, signal, window_size, offset_size):
    vals = []
    for i,j in enumerate(data[signal]):
        if(data.fs[i]>=1):
            window_sample =  int(data.fs[i]*window_size )
            offset =  int(data.fs[i]*offset_size)
            arr =  sliding_windows_variable(j, window_sample ,offset)
            vals.append(arr)
        else:
            vals.append('N/A')
    return vals
    
        
def compute_hf_windows(data, window_size):
    vals = []
    for i,j in enumerate(data.HF_IA):
        if(data.fs[i]>=1):
            arr =  sliding_windows(j, window_size)
            vals.append(arr)
        else:
            vals.append('N/A')
    return vals
    
def compute_mean_LF_IA(data, window_size, percent):
    window = compute_lf_windows(data, window_size)
    vals = []
    for k,interval in enumerate(window):
        vals_sub = []
        if(data.fs[k]>=1):
            for i in interval:
                arr_remove = remove_outliers(i,percent)
                vals_sub.append(np.mean(arr_remove))
            vals.append(np.array(vals_sub))
        else:
            vals.append('N/A')
    return vals
            
    
def compute_mean_HF_IA(data, window_size, percent):
    window = compute_hf_windows(data, window_size)
    vals = []
    for k,interval in enumerate(window):
        vals_sub = []
        if(data.fs[k]>=1):
            for i in interval:
                arr_remove = remove_outliers(i,percent)
                vals_sub.append(np.mean(arr_remove))
            vals.append(np.array(vals_sub))
        else:
            vals.append('N/A')
    return vals

def compute_mean_feature_IA(data, signal, window_size, offset_size, percent):
    window = compute_feature_windows(data, signal, window_size,offset_size)
    vals = []
    for k,interval in enumerate(window):
        vals_sub = []
        if(data.fs[k]>=1):
            for i in interval:
                arr_remove = remove_outliers(i,percent)
                vals_sub.append(np.mean(arr_remove))
            vals.append(np.array(vals_sub))
        else:
            vals.append('N/A')
    return vals
            
    
def sliding_windows(data, seq_length):
    x = []
    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        x.append(_x)

    return np.array(x)

def sliding_windows_variable(data, seq_length, offset):
    x = []
    for i in range(0, len(data)-seq_length-1, offset):
        _x = data[i:(i+seq_length)]
        x.append(_x)

    return np.array(x)    
        
def compute_sampling_rate(data):
    samples = data.samples
    duration = data.duration
    fs = samples/duration
    data['fs'] =  fs
    return

def compute_sampling_rate_array(dataframe,user,sections):
    data_sub = dataframe[dataframe.user==user]
    samples_total = 0
    duration_total = 0
    for i in sections:
        samples_total += data_sub[data_sub.section==i].samples.to_numpy()
        duration_total += data_sub[data_sub.section==i].duration.to_numpy()
    fs = samples_total/duration_total
        
    return fs, samples_total, duration_total

def construct_fs_all(dataframe, sections):
    arr_fs = []
    arr_duration = []
    arr_samples = []
    users =  dataframe.user.unique()
    for i in users:
        fs, samples_total, duration_total = compute_sampling_rate_array(dataframe, i, sections)
        arr_fs.append(fs)
        arr_samples.append(samples_total)
        arr_duration.append(duration_total)
    return arr_fs, arr_samples, arr_duration


def compute_windows_lf(data, window_size, offset_size):
    vals = []
    for i in range(len(data)):
        if (data.fs[i]>=1):
            window_sample =  int(data.fs[i]*window_size )
            offset =  int(data.fs[i]*offset_size)
            arr =  sliding_windows_variable(data.bandpass_lf_signal[i], window_sample, offset)
            vals.append(arr)
        else:
            vals.append('N/A')
    return vals

def compute_windows_hf(data, window_size, offset_size):
    vals = []
    for i in range(len(data)):
        if (data.fs[i]>=1):
            window_sample =  int(data.fs[i]*window_size )
            offset =  int(data.fs[i]* offset_size)
            arr =  sliding_windows_variable(data.bandpass_hf_signal[i], window_sample, offset)
            vals.append(arr)
        else:
            vals.append('N/A')
    return vals
    

def compute_mean_LF(data,  window_size, offset_size,  percent):
    window = compute_windows_lf(data, window_size, offset_size)
    vals = []
    for k,interval in enumerate(window):
        vals_sub = []
        if(data.fs[k]>=1):
            for i in interval:
                arr_remove = remove_outliers(i,percent)
                vals_sub.append(np.mean(arr_remove))
            vals.append(np.array(vals_sub))
        else:
            vals.append('N/A')
    return vals


def compute_mean_HF(data,  window_size, offset_size,  percent):
    window = compute_windows_hf(data, window_size, offset_size)
    vals = []
    for k,interval in enumerate(window):
        vals_sub = []
        if(data.fs[k]>=1):
            for i in interval:
                arr_remove = remove_outliers(i,percent)
                vals_sub.append(np.mean(arr_remove))
            vals.append(np.array(vals_sub))
        else:
            vals.append('N/A')
    return vals


def feature_per_user(dataframe, label_value, sample_random=5):
    featx = dataframe[dataframe.LF_IA_mean_signal!='N/A'].LF_IA_mean_signal.to_numpy()
    featy = dataframe[dataframe.HF_IA_mean_signal!='N/A'].HF_IA_mean_signal.to_numpy()
    arrx = np.array([])
    arry = np.array([])
    for i,j in zip(featx,featy):
        if sample_random and sample_random<i.shape[0]:
            idx =  np.random.choice(i.shape[0], int(sample_random), replace=False)  
            arrx, arry =  np.concatenate((arrx,i[idx].flatten())), np.concatenate((arry,j[idx].flatten()))
        else:
            arrx, arry =  np.concatenate((arrx,i.flatten())), np.concatenate((arry,j.flatten()))
    
    labels =  label_value*np.ones((len(arrx)))

        
    return arrx,arry,labels

    

def find_users(dataframe):
    users =  dataframe.user.unique()
    sections = dataframe.section.unique()
    users_accept = []
    for i in users:
        df = dataframe[dataframe.user==i]
        find_low_fs =  np.where(df.fs<1)[0].shape[0]
        if (find_low_fs==0):
            users_accept.append(i)
    return users_accept

def construct_signal_general(dataframe, users, sections):
    arr = []
    for i in users:
        vals = create_new_signal(dataframe, i, sections)
        arr.append(vals)
    return arr


def compute_hilbert_lf_ia_test(data,signal):
    vals = []
    for i,j in enumerate(data[signal]):
        if(data.fs[i]>=1):
            analytic_signal = hilbert(j)
            vals.append(np.abs(analytic_signal))
        else:
            vals.append('N/A')
    data['LF_IA_'+signal] = vals
    return


def compute_hilbert_hf_ia_test(data,signal):
    vals = []
    for i,j in enumerate(data[signal]):
        if(data.fs[i]>=1):
            analytic_signal = hilbert(j)
            vals.append(np.abs(analytic_signal))
        else:
            vals.append('N/A')
    data['HF_IA_'+signal] = vals
    return

def construct_fs_general(dataframe, users, sections):
    arr_fs = []
    arr_duration = []
    arr_samples = []
    for i in users:
        fs, samples_total, duration_total = compute_sampling_rate_array(dataframe, i, sections)
        arr_fs.append(fs)
        arr_samples.append(samples_total)
        arr_duration.append(duration_total)
    return arr_fs, arr_samples, arr_duration


def compute_labels(signal, val_label):
    labels = val_label*np.ones((len(signal)))
    return labels

def interpolate_signal(dataframe, num_points):
    dataframe['interpolate_signal']=""
    interp_signal = []
    for i , vals in enumerate(dataframe.signal.to_numpy()):
        x= len(vals)
        sample_points = np.linspace(0,x,num=num_points)
        spl = CubicSpline(np.linspace(0,x,num=x),vals)
        y = spl(sample_points)
        interp_signal.append(y)
        #dataframe.iloc[i].interpolate_signal =  y
    dataframe['interpolate_signal'] = interp_signal
    return

def interpolate_signal_all(dataframe):
    dataframe['interp_signal'] = " "
    signals = dataframe.signal
    interp_signal = []
    for i,vals in enumerate(signals):
        x= len(vals)
        sample_points = np.linspace(0,x,num=int(1.75*x))
        spl = CubicSpline(np.linspace(0,x,num=x),vals)
        y = spl(sample_points)
        interp_signal.append(y)
        #dataframe.signal[i] = y
        dataframe.samples[i] = int(1.75*x)
        dataframe.fs[i] =  int(1.75*x) / int(dataframe.duration[i])
        #dataframe.iloc[i].interpolate_signal =  y
    dataframe['interp_signal'] = interp_signal
    return

def compute_median_array(dataframe, section1, section2):
    lf_arr = dataframe[section1].to_numpy()
    hf_arr = dataframe[section2].to_numpy()

    dataframe['rest_median_lf'] = ""
    dataframe['rest_median_hf'] = ""
    median_lf = []
    median_hf = []
    for i,j in zip(lf_arr, hf_arr):
        median_lf.append(np.median(i))
        median_hf.append(np.median(j))
    dataframe['rest_median_lf'] =  median_lf
    dataframe['rest_median_hf'] =  median_hf

    return

    