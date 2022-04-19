import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import pickle as pkl
import utils_ as u

from TargetCnn import targetModel_cnn
from absl import app, flags

FLAGS = flags.FLAGS

def main(argv): 

    print("Load model")
    X_in, y_in, _, _ = pkl.load(open("Dataset/"+FLAGS.dataset_name+".pkl", "rb"))
    cnn_model = targetModel_cnn("BaseModel", FLAGS.window_size, FLAGS.channel_dim, FLAGS.class_nb, arch='1')    
    cnn_model.train([], new_train=False, checkpoint_path="TrainedModels/"+cnn_model.name)
    
    #Initializations
    adv_train = np.ndarray((0, 1, target.seg_size, target.channel_nb))
    clean_train = np.ndarray((0, 1, target.seg_size, target.channel_nb))
    y_adv_train = np.ndarray(0).astype(np.int)
    target_label = np.ndarray(0).astype(np.int)
    
    poly_transform = []
    for _ in range(FLAGS.CLASS_NB):
        poly_transform.append(tf.constant([]))
        
    #TSA-STAT Adversarial attack 
    for sample_id in range(len(X_in)):
    
        sys.stdout.write("\rDeg {} with target {} on Sample: {}/{}".format(FLAGS.deg, FLAGS.t, sample_id, X_in.shape[0]))
        sys.stdout.flush()
        
        X_s= X_in[sample_id:sample_id+1] #Original example
        y_s = y_in[sample_id] #Original label
        
        if y_s != t:
            clean_sample = tf.convert_to_tensor(X_s.reshape((1,1,SEG_SIZE,CHANNEL_NB)))
            y_orig = target.predict(clean_sample).numpy()[0]
            _, clean_stat_ft = u.m_features(clean_sample)
            proj_fn_dict = {'orig_X_ft':clean_stat_ft, 
                            'ft_set':tuple(FLAGS.ft_cons_set)}
                   
            Xadv, poly_transform[y_orig], _ = cnn_model.tsastat_attack(clean_sample, FLAGS.t, proj_fn_dict, a_init=poly_transform[y_orig],
                deg=FLAGS.deg, eta_init=5e-1, rho=FLAGS.rho, c=FLAGS.rho+1e-2, max_iter=5e3, verbose=False)  
        
            adv_train = np.concatenate([adv_train, Xadv.numpy().reshape(1, 1, target.seg_size, target.channel_nb)])
            clean_train = np.concatenate([clean_train, clean_sample.numpy().reshape(1, 1, target.seg_size, target.channel_nb)])
                
            y_adv_train = np.concatenate([y_adv_train, [y_s]])
            target_label = np.concatenate([target_label, [t]])
            pkl.dump([adv_train, clean_train, y_adv_train, target_label, poly_transform], open(f"TSASTAT_Attack_on_{FLAGS.dataset_name}_target_class_{FLAGS.t}.pkl", "wb"))
                
if __name__=="__main__":    

    flags.DEFINE_string('dataset_name', None, 'Dataset name')
    flags.DEFINE_integer('window_size', None, 'Window size of the input')
    flags.DEFINE_integer('channel_dim', None, 'Number of channels of the input')
    flags.DEFINE_integer('class_nb', None, 'Total number of classes')
    flags.DEFINE_integer('t', None, 'Attack target class label')
    flags.DEFINE_integer('deg', 0, 'Degree of the polynomial transformation')
    flags.DEFINE_integer('rho', -1, 'Confidence value for TSA-STAT attack')
    flags.DEFINE_enum('ft_cons_set', [0,1,2,3,4], 'Statistical features constraints')
    flags.mark_flags_as_required(['dataset_name', 'window_size', 'channel_dim', 'class_nb', 't'])
    app.run(main) 