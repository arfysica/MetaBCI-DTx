import numpy as np
from scipy.io import loadmat
import torch

data_path = "E:\DataSet\public_dataset\Base de Datos Habla Imaginada"

def load_imag_eeg_mat(sub, cls_list=[0,1,2,3,4]):
    """
    sub = 1, 2, 3, ..., 15

    F3 -- Muestra 1:4096
    F4 -- Muestra 4097:8192
    C3 -- Muestra 8193:12288
    C4 -- Muestra 12289:16384
    P3 -- Muestra 16385:20480
    P4 -- Muestra 20481:24576

    Etiquetas :  Modalidad: 1 - Imaginada
	     		            2 - Pronunciada
                             
    Estímulo:   1 - A
                2 - E
                3 - I
                4 - O
                5 - U
                6 - Arriba
                7 - Abajo
                8 - Adelante
                9 - Atrás
                10 - Derecha
                11 - Izquierda

    Artefactos: 1 - No presenta
			    2 - Presencia de parpadeo(blink)

    return  data.type = list
            len(data) = 11
            data[i].shape = (no.trials, channel, sample)
    """
    if sub < 10:
        sub = "\\S0" + str(sub)
    else:
        sub = "\\S" + str(sub)
    path = data_path + sub + sub + "_EEG"

    data = loadmat(path)
    eeg_data = data['EEG']

    raw_data = []
    for i in cls_list:
        data_i = []
        for j in range(eeg_data.shape[0]):
            x_j = eeg_data[j, :]
            if x_j[24577] == (i + 1) and x_j[24576] == 1:
                xi_j = x_j[0:24576]
                xi_j = xi_j.reshape(-1, 4096)
                data_i.append(xi_j)
        raw_data.append(np.array(data_i))
    return raw_data

def get_all_sub_data_list(cls_list=[0,1,2,3,4]):
    data = []
    for i in range(15):
        data_i = load_imag_eeg_mat(i+1, cls_list)
        data.append(data_i)
    return data

def down_sample_data(data):
    for i in range(15):
        for j in range(len(data[i])):
            d_ij = data[i][j]
            sampled_d_ij_list = [d_ij[:, :, i::4] for i in range(4)]
            data[i][j] = np.concatenate(sampled_d_ij_list, axis=0)
    return data

def gen_task_list(data):
    task_list = []
    for d_i in data:
        lb = [np.ones(d_i[j].shape[0]) * j for j in range(len(d_i))]
        data_i = np.concatenate(d_i, axis=0)
        lb_i = np.concatenate(lb)
        task_list.append((data_i, lb_i))
    return task_list

def maml_task_list(task_list, num_test_sub=3, random_seed=None):
    num_all_sub = len(task_list)
    if random_seed is not None:
        np.random.seed(random_seed)
    sub_list = np.arange(15)
    np.random.shuffle(sub_list) # 前num_test_sub个task作为测试task
    shuffled_task_list = [task_list[i] for i in sub_list]

    for i in range(num_all_sub):
        data, label = shuffled_task_list[i]

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)

        shuffled_data = data[indices]
        shuffled_label = label[indices]

        shuffled_task_list[i] = (shuffled_data, shuffled_label)

    tr_tasks = shuffled_task_list[num_test_sub:]
    te_tasks = shuffled_task_list[:num_test_sub]

    return tr_tasks, te_tasks

if __name__ == '__main__':
    data = get_all_sub_data_list()
    data = down_sample_data(data)
    # print([data[13][i].shape[0] for i in range(11)])
    # print(sum([data[13][i].shape[0] for i in range(11)]))
    task_list = gen_task_list(data)
    # print(len(task_list))
    # print(task_list[13][0].shape, task_list[13][1].shape)
    train_tasks, test_tasks = maml_task_list(task_list, num_test_sub=3, random_seed=42)
    print(len(train_tasks), len(test_tasks))
    print(train_tasks[0][0].shape, train_tasks[0][1].shape)