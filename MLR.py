from sklearn import linear_model
from preprocess import *
from sklearn.metrics import r2_score
import scipy.io
import numpy as np
import time

#读取数据
mat = mat = scipy.io.loadmat('data/test.mat')
EMG = mat['S1']
EEG = mat['S2']
freq_num = EEG.shape[0]
sample_num = EEG.shape[1]
ch_num = EEG.shape[2]

#参数
tap_size=1

X,Y = complex_data_chunking(EEG,EMG,sample_num,freq_num,ch_num,tap_size)

X=np.array([X.real,X.imag])
X=np.transpose(X,[1,0,2,3])
X=np.reshape(X,[sample_num-tap_size,2*tap_size*freq_num*32])
Y=np.array([Y.real,Y.imag])
Y=np.transpose(Y,[1,0,2])
Y=np.reshape(Y,[sample_num-tap_size,freq_num*5*2])

#分割训练集和测试集
n = int(sample_num*0.8)
X_train=X[0:n-1]
Y_train=Y[0:n-1]
X_test =X[n:-1]
Y_test =Y[n:-1]

tic=time.time()
model = linear_model.LinearRegression()
model.fit(X_train,Y_train)
toc=time.time()
print(toc-tic)

Y_hat=model.predict(X_test)
r2_val = r2_score(Y_test,Y_hat, multioutput='variance_weighted')
print(r2_val)

Y_test=np.reshape(Y_test,[X_test.shape[0],2,7,5])
Y_test=np.transpose(Y_test,[2,0,3,1])
Y_hat=np.reshape(Y_hat,[X_test.shape[0],2,7,5])
Y_hat=np.transpose(Y_hat,[2,0,3,1])

scipy.io.savemat('MLRtest.mat',{'actual':Y_test,'pred':Y_hat})



