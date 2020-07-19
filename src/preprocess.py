import numpy as np


def complex_data_chunking(eeg, emg, sample_num, freq_num, ch_num, tap_size):
    '''
    将输入数据切割成可用于训练的形式
    :param eeg: EEG信号经STFT的数据
    :param emg: EMG信号经STFT的数据
    :param sample_num: EEG、EMG信号的采样次数
    :param freq_num: STFT的频带个数
    :param ch_num: EEG信号的通道数
    :param tap_size:
    :return:
    '''
    emg_r = np.transpose(emg, (1, 0, 2))
    emg_r = emg_r.reshape((sample_num, 5 * freq_num))

    x = np.empty([(sample_num - tap_size), tap_size, ch_num * freq_num], dtype=complex)
    y = emg_r[tap_size - 1:sample_num - 1, :]
    for j in range(0, freq_num - 1):
        for i in range(tap_size + 1, x.shape[0]):
            x[i - tap_size - 1, :, j * ch_num:(j + 1) * ch_num] = eeg[j, i - tap_size - 1:i - 1, :]

    return x, y

def complex_data_dechunking(actual, pred, test_iterations, batch_size):
    '''
    把数据恢复为原格式
    :param actual:  实际EMG数据
    :param pred:    预测EMG数据
    :param test_iterations:     每轮的训练次数
    :param batch_size:  batch size
    :return:
    '''
    actual = actual.reshape([test_iterations * batch_size, 2, 7, 5])
    pred = pred.reshape([test_iterations * batch_size, 2, 7, 5])
    actual = np.transpose(actual, (2, 0, 3, 1))
    pred = np.transpose(pred, (2, 0, 3, 1))
    return actual,pred
