import os
import time
from cGRU_src.addition import R_squared
import numpy as np
import scipy.io
import tensorflow as tf
import custom_cells as cc
import GRU_wrapper as wg

# import state to state non-linearities
from cGRU_src.custom_cells import mod_relu

# import gate non-linearities
from custom_cells import gate_phase_hirose
from custom_cells import mod_sigmoid_prod
from custom_cells import mod_sigmoid_sum
from custom_cells import mod_sigmoid
from custom_cells import mod_sigmoid_beta
from custom_cells import mod_sigmoid_gamma
# from custom_cells import double_sigmoid

from custom_optimizers import RMSpropNatGrad
from synthetic_experiments import compute_parameter_total
from sklearn.metrics import r2_score
import logging
from preprocess import *
# 读取数据
mat = mat = scipy.io.loadmat('data/test_rGRU.mat')
EMG = mat['EMG']
EEG = mat['EEG']
time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
sample_num = EEG.shape[0]
EEG_ch_num = EEG.shape[1]
EMG_ch_num = EMG.shape[1]

# 参数
name='rGRU'
tap_size = 10
epoch = 5
n_units = 80  # 隐状态长度 hiden_size
learning_rate = 1e-3  # 学习率
decay = 0.9  # 衰减
batch_size = 256
GPU = 0  # GPU编号
activation = mod_relu  # 激活函数
cell_fun = cc.StiefelGatedRecurrentUnit  # RNN种类
gpu_mem_frac = 1.0  # GPU可用内存比例
qr_steps = -1
standardize = True #是否标准化
stiefel = True  # 是否使用stiefel优化
real = False    # 节点参数是否为实数
grad_clip = True   #是否进行 gradient clip
output_size = EMG_ch_num    #输出数据的大小
single_gate = False
gate_act_lst = [gate_phase_hirose, mod_sigmoid_prod, mod_sigmoid_sum,
                mod_sigmoid, mod_sigmoid_beta, mod_sigmoid_gamma]
gate_activation = gate_act_lst[0]
subfolder = name +'_'+'tapsize'+str(tap_size)+'batchsize'+str(batch_size)+'epoch'+str(epoch) # 结果存储的位置

#日志
if os.path.isdir('logs/'+subfolder)==False:
    os.mkdir('logs/'+subfolder)

if os.path.isdir('results/'+subfolder)==False:
    os.mkdir('results/' + subfolder)

if os.path.isdir('results/' + subfolder + '/'+time_str)==False:
    os.mkdir('results/' + subfolder + '/'+time_str)

if os.path.isdir('logs/' + subfolder + '/'+time_str)==False:
    os.mkdir('logs/' + subfolder + '/'+time_str)

logging.basicConfig(level=10,
			format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
             filename='logs/'+subfolder+'/'+time_str+'/log.log')
logging.info('name:'+str(name))
logging.info('tap_size:'+str(tap_size))
logging.info('epoch:'+str(epoch))
logging.info('n_units:'+str(n_units))
logging.info('learning_rate:'+str(learning_rate))
logging.info('batch_size:'+str(batch_size))
logging.info('activation:'+str(activation))

# 数据预处理
X,Y = real_data_chunking(EEG, EMG, sample_num, EEG_ch_num, tap_size)
del EEG
del EMG

#标准化
if standardize:
    Y_mean=np.mean(Y)
    Y_std=np.std(Y)
    X=(X-np.mean(X))/np.std(X)
    Y=(Y-np.mean(Y))/np.std(Y)

#分割训练集和测试集
n = int(sample_num*0.8)
X_train=X[0:n-1]
Y_train=Y[0:n-1]
X_test =X[n:-1]
Y_test =Y[n:-1]

train_iterations = int(X_train.shape[0] / batch_size)
test_iterations = int(X_test.shape[0] / batch_size)


# ------------------------- set up the rnn graph. ---------------------------
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    # #### Cell selection. ####
    if cell_fun.__name__ == 'UnitaryCell':
        cell = cell_fun(num_units=n_units, num_proj=output_size,
                        activation=activation, real=real)
    elif cell_fun.__name__ == 'StiefelGatedRecurrentUnit':
        cell = cell_fun(num_units=n_units, num_proj=output_size,
                        activation=activation, gate_activation=gate_activation,
                        stiefel=stiefel,
                        real=real, single_gate=single_gate,
                        complex_input=False)
    elif cell_fun.__name__ == 'GRUCell':
        cell = wg.RealGRUWrapper(cell_fun(num_units=n_units), output_size)
    else:
        cell = cell_fun(num_units=n_units, num_proj=output_size,
                        use_peepholes=True)
    # #### Input & Output ####
    x = tf.placeholder(tf.float32, shape=(batch_size, tap_size, EEG_ch_num), name='EEG')
    y = tf.placeholder(tf.float32, shape=(batch_size, EMG_ch_num), name='EMG')


    y_hat = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    y_hat = y_hat[0]  # throw away the final state.
    y_hat = y_hat[:, -1,:]  # only the final output is interesting.
    loss = tf.losses.mean_squared_error(y, y_hat)
    R2= R_squared(y, y_hat)
    tf.summary.scalar('mse', loss)
    tf.summary.scalar('R2', R2)

    optimizer = RMSpropNatGrad(learning_rate=learning_rate, decay=decay,
                               global_step=global_step, qr_steps=qr_steps)
    if grad_clip:
        with tf.variable_scope("gradient_clipping"):
            gvs = optimizer.compute_gradients(-R2)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    else:
        train_op = optimizer.minimize(-R2, global_step=global_step)
    init_op = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()
    parameter_total = compute_parameter_total(tf.trainable_variables())



# choose the GPU to use and how much memory we require.
gpu_options = tf.GPUOptions(visible_device_list=str(GPU),
                            per_process_gpu_memory_fraction=gpu_mem_frac)
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)

summary_writer = tf.summary.FileWriter('logs' + '/' + subfolder + '/'   + time_str, graph=graph)
# ------------------------- and run it! ---------------------------------
train_plot = []
print('total iteration',str(epoch*train_iterations))
with tf.Session(graph=graph, config=config) as sess:
    init_op.run()
    for j in range(0,epoch):
        for i in range(train_iterations):
            x_batch = X_train[(i) * batch_size:(i + 1) * batch_size, :, :]
            y_batch = Y_train[(i) * batch_size:(i + 1) * batch_size]
            feed_dict ={x:x_batch,
                        y:y_batch}
            run_lst = [loss, summary_op, global_step, y_hat,R2,train_op ]
            tic = time.time()
            np_loss_train, summary_mem, np_global_step, Y_hat,r2,_ = \
                sess.run(run_lst, feed_dict=feed_dict)
            toc = time.time()

            r2_val = r2_score(y_batch, Y_hat)

            if i % 25 == 0:
                print('iteration', (i+j*train_iterations) / 100, '*10^2',
                      np.array2string(np.array(r2),
                                      precision=4),
                      )
            train_plot.append([i / 100, np_loss_train, r2])
            summary_writer.add_summary(summary_mem, global_step=np_global_step)

        saver = tf.train.Saver()
        saver.save(sess,"logs/"+subfolder+'/'   + time_str +"/checkpoint_dir/epoch"+str(j+1))#b保存

    test_losses = []
    r2_vals = []
    actual=np.empty([test_iterations * batch_size, EMG_ch_num], dtype='float32')
    pred=np.empty([test_iterations * batch_size, EMG_ch_num], dtype='float32')
    for i in range(test_iterations):
        x_batch = X_train[(i) * batch_size:(i + 1) * batch_size, :, :]
        y_batch = Y_train[(i) * batch_size:(i + 1) * batch_size]
        feed_dict ={x:x_batch,
                    y:y_batch}
        run_lst = [loss, summary_op, global_step, train_op]
        np_loss_test,Y_hat,r2 = sess.run([loss,y_hat,R2], feed_dict=feed_dict)
        actual[i * batch_size:((i + 1) * batch_size), :]=y_batch
        pred[i * batch_size:((i + 1) * batch_size), :]=Y_hat
        print('R2_'+str(i),':',r2)
        test_losses.append(np_loss_test)
        r2_vals.append(r2)

    #输出结果
    print('test loss', np.mean(test_losses),'R2:',np.mean(r2_vals))
    r2_val = r2_score(actual, pred, multioutput='variance_weighted')
    print('all R2', r2_val)
    logging.info('all R2:'+str(r2_val))


    #从标准化变回
    if standardize:
        actual = actual*Y_std+Y_mean
        pred = pred * Y_std + Y_mean

    r2_val=np.empty([5])
    for i in range(EMG_ch_num):
        r2_val[i]=r2_score(actual[:,i], pred[:,i], multioutput='variance_weighted')
    print('R2 for each channel:',r2_val)

    #保存结果
    scipy.io.savemat('results/'+subfolder+'/'+time_str+'/'+subfolder+'.mat',{'actual':actual,'pred':pred})
summary_writer.close()
