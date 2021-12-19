import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


twins_t = pd.read_csv('/content/twin_pairs_T_3years_samesex.csv')
twins_x = pd.read_csv('/content/twin_pairs_X_3years_samesex.csv')
twins_y = pd.read_csv('/content/twin_pairs_Y_3years_samesex.csv')
twins = pd.concat((twins_t,twins_y,twins_x),axis=1).drop(['Unnamed: 0','Unnamed: 0.1','infant_id_0','infant_id_1'],axis=1).dropna(axis=0)
twins = twins[(twins.dbirwt_0<=2000)&(twins.dbirwt_1<=2000)]

#######################################################################
# may need to run several times to find a good treatment distribution #
#######################################################################
twins_x = twins.iloc[:,4:]
# follow the pre-proccessing steps in CEVAE (https://arxiv.org/pdf/1705.08821.pdf)
W_o = np.random.normal(scale=0.1,size=(twins_x.shape[1]-1,1))
W_h = np.random.normal(loc=5,scale=0.1)
tmp = np.matmul(twins_x.drop('gestat10',axis=1).values,W_o) + W_h*(twins_x['gestat10'].values/10-0.1).reshape(-1,1)
p = 1/(1 + np.exp(-tmp))
twins['t'] = np.random.binomial(1,p)
# twins_bord = twins[['bord_0','bord_1']]
twins['yf'] = twins[['mort_0','mort_1','t']].apply(lambda x: x[int(x[2])],axis=1)
twins['ycf'] = twins[['mort_0','mort_1','t']].apply(lambda x: x[int(1-x[2])],axis=1)
# my extra steps: create 'bord' and drop 'bord_0','bord_1'
twins['bord'] = twins[['bord_0','bord_1','t']].apply(lambda x: x[int(x[2])],axis=1)
twins.drop(['dbirwt_0','dbirwt_1','mort_0','mort_1','bord_0','bord_1'],axis=1,inplace=True)
twins = twins.sample(frac=1).reset_index(drop=True) # shuffle the rows

n_splits = 5

twins_1_5_x_train = np.zeros((726,49,n_splits))
twins_1_5_t_train = np.zeros((726,n_splits))
twins_1_5_yf_train = np.zeros((726,n_splits))
twins_1_5_ycf_train = np.zeros((726,n_splits))
twins_1_5_mu0_train = np.zeros((726,n_splits))
twins_1_5_mu1_train = np.zeros((726,n_splits))

twins_1_5_x_test = np.zeros((243,49,n_splits))
twins_1_5_t_test = np.zeros((243,n_splits))
twins_1_5_yf_test = np.zeros((243,n_splits))
twins_1_5_ycf_test = np.zeros((243,n_splits))
twins_1_5_mu0_test = np.zeros((243,n_splits))
twins_1_5_mu1_test = np.zeros((243,n_splits))

for idx, arr in enumerate(np.array_split(twins,5)):
    train_set, test_set = train_test_split(arr,stratify=arr.t)
    train_set, test_set = train_set.iloc[:726,:], test_set.iloc[:243,:]

    twins_1_5_x_train[:,:,idx] = train_set.drop(['t','yf','ycf'],axis=1)
    twins_1_5_t_train[:,idx] = train_set['t']
    twins_1_5_yf_train[:,idx] = train_set['yf']
    twins_1_5_ycf_train[:,idx] = train_set['ycf']
    twins_1_5_mu0_train[:,idx] = twins_1_5_yf_train[:,idx]*(1-twins_1_5_t_train[:,idx]) + twins_1_5_ycf_train[:,idx]*twins_1_5_t_train[:,idx]
    twins_1_5_mu1_train[:,idx] = twins_1_5_yf_train[:,idx]*twins_1_5_t_train[:,idx] + twins_1_5_ycf_train[:,idx]*(1-twins_1_5_t_train[:,idx])

    twins_1_5_x_test[:,:,idx] = test_set.drop(['t','yf','ycf'],axis=1)
    twins_1_5_t_test[:,idx] = test_set['t']
    twins_1_5_yf_test[:,idx] = test_set['yf']
    twins_1_5_ycf_test[:,idx] = test_set['ycf']
    twins_1_5_mu0_test[:,idx] = twins_1_5_yf_test[:,idx]*(1-twins_1_5_t_test[:,idx]) + twins_1_5_ycf_test[:,idx]*twins_1_5_t_test[:,idx]
    twins_1_5_mu1_test[:,idx] = twins_1_5_yf_test[:,idx]*twins_1_5_t_test[:,idx] + twins_1_5_ycf_test[:,idx]*(1-twins_1_5_t_test[:,idx])

np.savez('twins_1-5_train',
         x=twins_1_5_x_train,
         t=twins_1_5_t_train,
         yf=twins_1_5_yf_train,
         ycf=twins_1_5_ycf_train,
         mu0=twins_1_5_mu0_train,
         mu1=twins_1_5_mu1_train)
np.savez('twins_1-5_test',
         x=twins_1_5_x_test,
         t=twins_1_5_t_test,
         yf=twins_1_5_yf_test,
         ycf=twins_1_5_ycf_test,
         mu0=twins_1_5_mu0_test,
         mu1=twins_1_5_mu1_test)