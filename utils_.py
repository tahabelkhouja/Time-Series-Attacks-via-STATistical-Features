import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import pickle as pkl
import numpy.linalg as lg
import scipy.stats as st


def m_features(s):
    """
    Parameters
    ----------
    s : Input Signal 1x1xSxC

    Returns
    -------
    m_res_dict : Statistical features name dict
    m_res : Array of Statistial features values of Input signal

    """
    channel_nb = s.shape[-1]
    l = s.shape[2]
    #Stack init using channel 0
    
    #Mean
    m = tf.reduce_mean(s[0,0,:,0])
    
    #Median and Interquartile range
    cut_points = tfp.stats.quantiles(s[0,0,:,0], num_quantiles=4, 
                                     interpolation='midpoint')
    median = cut_points[2]
    Q1 = cut_points[1]
    Q3 = cut_points[3]
    quart_range = Q3-Q1 
    
    #Std deviation
    std_dev = tfp.stats.stddev(s[0,0,:,0])
    
    #Skewness
    skw1 = tf.add(Q1, m)
    skw2 = tf.subtract(skw1, tf.multiply(tf.cast(2.0, dtype=tf.float64), m))
    skew = tf.divide(skw2, quart_range)
    
    #Kurtosis
    kurt_1 = tf.reduce_sum(tf.pow(tf.subtract(s[0,0,:,0], m), 4))
    kurt_2 = tf.multiply(tf.cast(tf.divide(1.0, l), dtype=tf.float64), kurt_1)
    kurt_3 = tf.divide(kurt_2, tf.pow(std_dev, 4))
    kurt = tf.subtract(kurt_3, 3)
    
    #RMS
    rms_1 = tf.reduce_sum(tf.pow(s[0,0,:,0], 2))
    rms_2 = tf.multiply(tf.cast(tf.divide(1.0, l), dtype=tf.float64), rms_1)
    rms = tf.sqrt(rms_2)
    
    
    m_res = tf.reshape(tf.stack([m, std_dev, skew, kurt, rms]), (5,1))
    #Multiple channel case
    if channel_nb > 1:
        for c in range(1, channel_nb):
            #Mean
            m = tf.reduce_mean(s[0,0,:,c])
            
            #Median and Interquartile range
            cut_points = tfp.stats.quantiles(s[0,0,:,c], num_quantiles=4, 
                                             interpolation='midpoint')
            median = cut_points[2]
            Q1 = cut_points[1]
            Q3 = cut_points[3]
            quart_range = Q3-Q1 
            
            #Std deviation
            std_dev = tfp.stats.stddev(s[0,0,:,c])
            
            #Skewness
            skw1 = tf.add(Q1, m)
            skw2 = tf.subtract(skw1, tf.multiply(tf.cast(2.0, dtype=tf.float64), m))
            skew = tf.divide(skw2, quart_range)
            
            #Kurtosis
            kurt_1 = tf.reduce_sum(tf.pow(tf.subtract(s[0,0,:,c], m), 4))
            kurt_2 = tf.multiply(tf.cast(tf.divide(1.0, l), dtype=tf.float64), kurt_1)
            kurt_3 = tf.divide(kurt_2, tf.pow(std_dev, 4))
            kurt = tf.subtract(kurt_3, 3)
            
            #RMS
            rms_1 = tf.reduce_sum(tf.pow(s[0,0,:,c], 2))
            rms_2 = tf.multiply(tf.cast(tf.divide(1.0, l), dtype=tf.float64), rms_1)
            rms = tf.sqrt(rms_2)
            
            m_res = tf.concat([m_res, tf.reshape([m, std_dev, skew, kurt, rms], (5,-1))], axis=1)
    m_res_dict={'mean':0,
                'std_dev':1,
                'skew':3,
                'kurt':4,
                'rms':5
                }
    
    return m_res_dict, m_res

def mode(s):
    nbins = 5000
    range_mod = [tf.reduce_min(s[:,:,:,0]), tf.reduce_max(s[:,:,:,0])]
    hist_mod = tf.histogram_fixed_width_bins(s[:,:,:,0], range_mod, nbins=nbins)
    a,b, count = tf.unique_with_counts(tf.reshape(hist_mod, [-1]))
    idx_mod = a[tf.argmax(count)]
    alpha = tf.subtract(tf.reduce_max(s[:,:,:,0]), tf.reduce_min(s[:,:,:,0]))
    denum = tf.multiply(alpha, tf.dtypes.cast(tf.add(tf.multiply(2, idx_mod), 1), tf.float64))
    mode_ = tf.divide(denum, tf.dtypes.cast(tf.multiply(2, nbins), tf.float64))
    mode = tf.add(mode_, tf.reduce_min(s[:,:,:,0]))
    return mode


def ft_stat_score(x1, x2):
    _, s1 = m_features(x1)
    _, s2 = m_features(x2)
    return np.linalg.norm(s2-s1, ord=2)


        
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    