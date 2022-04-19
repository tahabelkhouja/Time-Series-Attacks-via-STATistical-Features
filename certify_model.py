import pickle as pkl

from TargetCnn import targetModel_cnn
from Certification import certifiedBound

from absl import app, flags

FLAGS = flags.FLAGS

def main(argv): 
   
    print("Load model")
    _, _, X_test, _ = pkl.load(open("Dataset/"+FLAGS.dataset_name+".pkl", "rb"))
    cnn_model = targetModel_cnn("BaseModel", FLAGS.window_size, FLAGS.channel_dim, FLAGS.class_nb, arch='1')    
    cnn_model.train([], new_train=False, checkpoint_path="TrainedModels/"+cnn_model.name)
   

    cetif_vect = []
    for sample_index in range(X_test.shape[0]):
        sys.stdout.write("\r{}: Iteration {}/{} . . . \n".format(cnn_model.name, 
                                           sample_index+1, X_test.shape[0]))
        sys.stdout.flush()
        x = X_test[sample_index:sample_index+1]
        orig_pred = cnn_model.predict(x).numpy()[0]
        _, _, L = certifiedBound(cnn_model, x, FLAGS.class_nb, FLAGS.mu_p, FLAGS.sigma, FALGS.iter_max)
        cetif_vect.extend(L)

    pkl.dump(cetif_vect, open(f"{cnn_model.name}_Certificates_{FLAGS.mu_n}_{FLAGS.sigma}.pkl", "wb"))


if __name__=="__main__":    

    flags.DEFINE_string('dataset_name', None, 'Dataset name')
    flags.DEFINE_integer('window_size', None, 'Window size of the input')
    flags.DEFINE_integer('channel_dim', None, 'Number of channels of the input')
    flags.DEFINE_integer('class_nb', None, 'Total number of classes')
    flags.DEFINE_integer('iter_max', 500, 'Maximum number of iterations')
    flags.DEFINE_float('mu_p', 1e-2, 'Certification mean mu_p')
    flags.DEFINE_float('sigma', 1e-1, 'Certification covariance')
    flags.mark_flags_as_required(['dataset_name', 'window_size', 'channel_dim', 'class_nb'])
    app.run(main) 













