import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import theano
import theano.tensor as tt
import seaborn as sns
import time

print('Running on PyMC3 v{}'.format(pm.__version__))
path = "C:/Users/ChoHeeseung/Desktop/BART/Uijong_New/People" ##Set path your record files

def dt_Participant(N, path):
    file = path+"/P{}/Record.txt".format(N)
    tmp = pd.read_csv(file, ' ', header = None)
    df = tmp[[4,8]].replace('Fail',0).astype(int)
    df.columns = ['pumps','pop_point']
    df['earning'] = df['pumps']*2   
    for i in range(len(df)):
        if df['pumps'][i] == 0:
            df['pumps'][i] = df['pop_point'][i]
    df['poped'] = df['pumps'] == df['pop_point']
    return df
    
def BernData(N, Max_pump):
    data = dt_UijongParticipant(N)['pumps']
    new_data = []
    for pumps in data:
        p = np.concatenate((np.ones(pumps), np.zeros(Max_pump-pumps)), axis = 0)
        new_data.append(p)
    return np.array(new_data)

def 2Parameter_Model(N_people, SEED, Max_pump = 128)
    '''
    N_people: The Number of participants you have.
    SEED: Set radnom seed of MCMC sampling.
    Max_pump: Maxiumum trial of your BART setting. My setting is 128.
    '''
    traces_gamma = []
    traces_beta = []
    for part in range(1,N_people):
        with pm.Model() as total_model:
            start_time = time.time()        
            p = 0.15    
            obs = BernData(part)
            #Pth participant            
            gamma_plus = pm.Uniform("gamma_plus",0,10) 
            beta = pm.Uniform("beta",0,10)
            omega_k = -gamma_plus/np.log(1-p)        
            for i in range(len(obs)):
                for l in range(Max_pump):
                    theta_lk = 1/(1+np.exp(beta*(l - omega_k)))
                    prob = pm.Bernoulli("prob_{}_{}".format(i,l), p=theta_lk, observed = obs[i][l])        
            _trace = pm.sample_smc(1000, cores = 6, random_seed = SEED)
            print("Sampling end:",part,"--- %s seconds ---" % (time.time() - start_time))          
        traces_gamma.append(_trace["gamma_plus"])
        traces_beta.append(_trace["beta"])
    return traces_gamma, traces_beta
