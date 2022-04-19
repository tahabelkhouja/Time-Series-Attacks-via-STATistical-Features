import numpy as np
from scipy.stats import random_correlation


def random_eig_vect(n):
    '''
    Get a random vector of size n with a sum det
    '''
    vec_res = []
    for _ in range(n-1):
        vec_res.append(np.random.uniform(low=0,high=1))
    if np.sum(vec_res) >=n:
        vec_res = [v-np.random.uniform(np.min(vec_res)) for v in vec_res]
    vec_res.append(n-np.sum(vec_res))
    return vec_res
    
def random_MVGuaussian_params(max_mu, std, n):
    '''
    Get random params of an MVG
    '''
    mu = np.random.uniform(0,max_mu, size=n)
    semi_pos = False
    while not semi_pos:
        corr =  random_correlation.rvs(tuple(random_eig_vect(n)))
        cov = np.dot(corr, np.diag(std*np.ones(n)))
        if np.sum(np.linalg.eigvals(cov)>=0)==n:
                semi_pos = True
    return mu, cov

def certifiedBound(target, x, class_nb, delta_mu, std_n, n_iter):
    pred_class = target.predict(x)[0]
    p_ns = np.zeros(class_nb)
    q_ns = np.zeros(class_nb)
    pred_distribution = []
    q_distribution = []
    for _ in range(10):
        if x.shape[-1]==1:
            mu = np.zeros(1)
            sigma = std_n * np.ones((1,1))
        else:
            mu, sigma = random_MVGuaussian_params(0, std_n, x.shape[-1])
        for _ in range(int(n_iter/10)):
            n1 = np.random.multivariate_normal(mu+delta_mu, sigma, x.shape[-2])
            n1 -= np.mean(n1)-delta_mu
            n2 = np.random.multivariate_normal(mu, sigma, x.shape[-2])
            n2_mean = np.mean(n2)
            n2 -= n2_mean
            x1 = x + n1[np.newaxis, np.newaxis, :,:]
            x2 = x + n2[np.newaxis, np.newaxis, :,:]
            pi = target.predict(x1)[0]
            qi = target.predict(x2)[0]
            pred_distribution.append(pi)
            q_distribution.append(qi)
    pred_distribution = np.array(pred_distribution)
    q_distribution = np.array(q_distribution)
    for c in range(class_nb):
        p_ns[c] = (np.sum(pred_distribution==c))/len(pred_distribution)
        q_ns[c] = (np.sum(q_distribution==c))/len(q_distribution)
    p = sorted(p_ns, reverse=True)
    if np.argmax(p_ns)!=pred_class :
        return np.argmax(p_ns), np.argmax(q_ns), -10 #-10 is non-certifiable
    if p[1]==0 :
        return np.argmax(p_ns), np.argmax(q_ns), -1 #-1 is robust
    
    #certification case
    bound_funct = lambda alpha:(np.sqrt( -(2/(np.sum(sigma)*alpha)) * \
                      np.log(1-p[0]-p[1] + 2 * \
                             np.power((0.5 * (np.power(p[0], 1-alpha)+ np.power(p[1], 1-alpha))), (1/(1-alpha)) )\
                             ) ) ) 
    x_range = np.hstack([np.arange(0.1, 1, 1e-5),np.arange(1+1e-5, 10, 1e-5)])
    delta_mu = np.max(bound_funct(x_range))
    return np.argmax(p_ns), np.argmax(q_ns), delta_mu
    
    

