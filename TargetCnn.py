import sys

import numpy as np
import tensorflow as tf
import utils_ as u
    
#CNN Architecture
class targetModel_cnn():
    def __init__(self, name, seg_size, channel_nb, class_nb, arch='1'):
        self.name = name
        self.seg_size = seg_size
        self.channel_nb = channel_nb
        self.class_nb = class_nb
        self.x_holder = []
        self.y_holder = []
        self.y_ =[]
        
        if arch=='0':
            self.trunk_model = tf.keras.Sequential([
                #Layers
                tf.keras.layers.Conv2D(20,[1, 12], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 2), strides=2),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.BatchNormalization(),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
        if arch=='1':
            self.trunk_model = tf.keras.Sequential([
                #Layers
                tf.keras.layers.Conv2D(66,[1, 12], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=4),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.15),
                tf.keras.layers.BatchNormalization(),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
        elif arch=='2':
            self.trunk_model = tf.keras.Sequential([ 
                #Layers
                tf.keras.layers.Conv2D(100,[1, 12], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=1),
                tf.keras.layers.Conv2D(50,[1, 5], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 4), strides=1),
                tf.keras.layers.Conv2D(50,[1, 3], padding="same", input_shape=(1, self.seg_size, self.channel_nb)),
                tf.keras.layers.MaxPooling2D((1, 2), strides=1),
                #Fully connected layer
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(200, activation=tf.nn.relu),
                tf.keras.layers.Dense(100, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.BatchNormalization(),
                #Logits layer
                tf.keras.layers.Dense(self.class_nb)
                ])
        self.model = tf.keras.Sequential([self.trunk_model,
            tf.keras.layers.Softmax()])
        #Training Functions
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

                
            
    
    def train(self, train_set, checkpoint_path="TrainingRes/model_target", epochs=10, new_train=False):
        
        @tf.function
        def train_step(X, y):
            with tf.GradientTape() as tape: 
                pred = self.model(X, training=True)
                pred_loss = self.loss_fn(y, pred)
                total_loss = pred_loss 
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            #tf.print("---Current Loss:", total_loss)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
        if not new_train:
            self.model.load_weights(checkpoint_path)
            print("\nWeights loaded!")
        else:
            for ep in range(epochs):
                sys.stdout.write("\r{}: Epochs {}/{} . . .".format(self.name, ep+1, epochs))
                sys.stdout.flush()
                for X, y in train_set:
                    train_step(X, y)
                self.model.save_weights(checkpoint_path)
            sys.stdout.write("\n")
                
    
    def predict(self, X):
        return tf.argmax(self.model(X, training=False), 1)
        
    def predict_stmax(self, X):
        return self.trunk_model(X, training=False)
    
    def score(self, X, y):
        X = tf.cast(X, tf.float64)
        acc = tf.keras.metrics.Accuracy()
        acc.reset_states()
        pred = self.predict(X)
        acc.update_state(pred, y)
        return acc.result().numpy()
    
    
    @tf.function
    def adv_loss_fn(self, X, t, rho):
        """
        Equation 4.3 in the main paper
        """
        Z_logits = tf.reshape(self.trunk_model(X, training=False), (self.class_nb,))
        return tf.cast(tf.maximum(tf.maximum(tf.reduce_max(Z_logits[:t]), 
                                          tf.reduce_max(Z_logits[t+1:]))-Z_logits[t], rho), dtype=tf.float64)
    
    def tsastat_attack(self, X, t, proj_fn_dict, a_init=[], deg=1, 
                       eta_init=1e-2,  c=-1e-2, rho=-20.0, max_iter=1e5, 
                       clip=None, channel=None,
                       verbose=False):
        """
        TSA-STAT main attack function
        X: Clean Sample.
        t: Target class label of the attack.
        proj_fn: Projection function to be used
        proj_fn_dict: Dict parameters of proj_fn
        :retrun: Adversarial example of X.
        """
        def lr_decay(epoch, min_lr=1e-4):
        	drop = 0.5
        	epochs_drop = 100
        	lrate = eta_init * np.power(drop, np.floor((epoch)/epochs_drop))
        	return max(lrate, min_lr) 
        
        @tf.function
        def TF(X, a, deg=5, clip=None, channel=None, mask=None): 
            """
            TF is is the polynomial transformation TF(.) presented in the main paper
            """
            if channel:
                assert deg==0, "Limited channels only supports degree 0"
                assert mask is not None, "Mask needs to be defined explicitly"
                X_res = tf.identity(X)
                mask = tf.convert_to_tensor(mask, dtype=tf.float64)
                for i, ch in enumerate(channel):
                    X_res = tf.add(X_res, tf.multiply(a, mask))
                if clip:
                    return tf.clip_by_value(X_res, clip[0], clip[1])
                else:
                    return X_res
                
            
            if deg==0:
                if clip:
                    return tf.clip_by_value(tf.add(X, a), clip[0], clip[1])
                else:
                    return tf.add(X, a)
            else:
                res = tf.zeros_like(X)
                for d in range(deg, -1, -1):
                    b = tf.pow(X, d)
                    ax = tf.multiply(a[d], b)
                    res = tf.add(res, ax)
                if clip:
                    return tf.clip_by_value(res, clip[0], clip[1])
                else:
                    return res
        
        @tf.function
        def loss_fct(X, ft_set, orig_X_ft): 
            """
            Equation 4.4 in the main paper
            Inputs:
                X: Input
                ft_set: Index of statistical features used according to utils_.m_features
                orig_X_ft: Values of the statiscal feature of a reference input
            """
            _, stats_ft = u.m_features(X)
            c_factor = tf.ones(stats_ft.shape[0], dtype=tf.float64)
            loss_ = tf.multiply(c_factor[ft_set[0]], tf.norm((stats_ft[ft_set[0]] - orig_X_ft[ft_set[0]]), ord=np.inf))
            for k in ft_set[1:]:
                loss_k = tf.multiply(c_factor[k], tf.norm((stats_ft[k] - orig_X_ft[k]), ord=np.inf))
                loss_ = tf.add(loss_, loss_k)
            return loss_

        
        if a_init.shape == 0:
            a = tf.random.uniform([deg+1, X.shape[-2], X.shape[-1]], minval=-1, maxval=1,  dtype=tf.float64)
        else:
            a = a_init
            
        min_a = tf.identity(a)
        lbda = tf.Variable(1.0, dtype=tf.float64) 
        loss_ = np.inf
        min_loss = np.inf
        ep = 0
        
        while ( ep <= max_iter):
            ep += 1
            eta = lr_decay(ep)
            with tf.GradientTape() as tape: 
                tape.watch(a)
                loss1 = self.adv_loss_fn(TF(X,a, deg=deg, clip=clip), t, rho)
                loss2 = loss_fct(TF(X,a, deg=deg, clip=clip), **proj_fn_dict)
                loss_ = tf.add(loss1, tf.multiply(lbda, loss2))
                if loss_ <= c:
                    return TF(X, a, deg=deg, clip=clip), a, [loss1, loss2]
                if loss_ < min_loss:
                    min_loss = loss_
                    min_a = tf.identity(a)
                grad = tape.gradient(loss_, a)
                if (ep%100==0) and verbose: print("Epoch{}:\nLoss1: {:.4f}\nLoss2: {:.4f}\nLoss: {:.4f}\n---".format(ep,loss1, loss2, loss_))
                a = a - eta * grad
        return TF(X, min_a, deg=deg, clip=clip), min_a,\
                [self.adv_loss_fn(TF(X,min_a, deg=deg, clip=clip), t, rho),\
                 loss_fct(TF(X, min_a, deg=deg, clip=clip), **proj_fn_dict)]
    
    

        
             