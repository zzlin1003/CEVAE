import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

twins_t = pd.read_csv('/content/twin_pairs_T_3years_samesex.csv')
twins_x = pd.read_csv('/content/twin_pairs_X_3years_samesex.csv')
twins_y = pd.read_csv('/content/twin_pairs_Y_3years_samesex.csv')
twins = pd.concat((twins_t,twins_y,twins_x),axis=1).drop(['Unnamed: 0','Unnamed: 0.1','infant_id_0','infant_id_1'],axis=1).dropna(axis=0)
twins = twins[(twins.dbirwt_0<=2000)&(twins.dbirwt_1<=2000)]

# follow the pre-proccessing steps of SITE (https://par.nsf.gov/servlets/purl/10123149)
w = np.random.uniform(low=-0.1,high=0.1,size=(twins_x.shape[1]-2,1))
n = np.random.normal(scale=0.1, size=(twins_x.shape[0],1))
tmp = np.matmul(twins_x.drop(['bord_0','bord_1'],axis=1).values,w) + n
p = 1/(1 + np.exp(-tmp))
twins['t'] = np.random.binomial(1,p)

twins['yf'] = twins[['mort_0','mort_1','t']].apply(lambda x: x[int(x[2])],axis=1)
twins['ycf'] = twins[['mort_0','mort_1','t']].apply(lambda x: x[int(1-x[2])],axis=1)
twins['bord'] = twins[['bord_0','bord_1','t']].apply(lambda x: x[int(x[2])],axis=1)
twins.drop(['dbirwt_0','dbirwt_1','mort_0','mort_1','bord_0','bord_1'],axis=1,inplace=True)
twins = twins.sample(frac=1).reset_index(drop=True) # shuffle rows

train_set, test_set = train_test_split(twins, stratify=twins.t)
nrows_train, nrows_test = train_set.shape[0], test_set.shape[0]

twins_1_1_x_train = np.expand_dims(train_set.drop(['t','yf','ycf'],axis=1).values,axis=2)
twins_1_1_t_train = np.expand_dims(train_set.t.values,axis=1)
twins_1_1_yf_train = np.expand_dims(train_set.yf.values,axis=1)
twins_1_1_ycf_train = np.expand_dims(train_set.ycf.values,axis=1)
twins_1_1_mu0_train = twins_1_1_ycf_train*twins_1_1_t_train + twins_1_1_yf_train*(1-twins_1_1_t_train)
twins_1_1_mu1_train = twins_1_1_yf_train*twins_1_1_t_train + twins_1_1_ycf_train*(1-twins_1_1_t_train)

twins_1_1_x_test = np.expand_dims(test_set.drop(['t','yf','ycf'],axis=1).values,axis=2)
twins_1_1_t_test = np.expand_dims(test_set.t.values,axis=1)
twins_1_1_yf_test = np.expand_dims(test_set.yf.values,axis=1)
twins_1_1_ycf_test = np.expand_dims(test_set.ycf.values,axis=1)
twins_1_1_mu0_test = twins_1_1_ycf_test*twins_1_1_t_test + twins_1_1_yf_test*(1-twins_1_1_t_test)
twins_1_1_mu1_test = twins_1_1_yf_test*twins_1_1_t_test + twins_1_1_ycf_test*(1-twins_1_1_t_test)

np.savez('twins_1-1_train',
         x=twins_1_1_x_train,
         t=twins_1_1_t_train,
         yf=twins_1_1_yf_train,
         ycf=twins_1_1_ycf_train,
         mu0=twins_1_1_mu0_train,
         mu1=twins_1_1_mu1_train)
np.savez('twins_1-1_test',
         x=twins_1_1_x_test,
         t=twins_1_1_t_test,
         yf=twins_1_1_yf_test,
         ycf=twins_1_1_ycf_test,
         mu0=twins_1_1_mu0_test,
         mu1=twins_1_1_mu1_test)