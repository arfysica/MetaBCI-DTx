import torch
import torch.optim as optim
import torch.nn as nn
from datetime import date
from maml import MAML
from utils import support_query_split
import numpy as np
from load_Base_de_Datos_Habla_Imaginada import get_all_sub_data_list, down_sample_data, gen_task_list, maml_task_list
from eegnet import EEGNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = get_all_sub_data_list(cls_list=[0,1,2,3,4])
data = down_sample_data(data)
task_list = gen_task_list(data)
train_tasks, test_tasks = maml_task_list(task_list, num_test_sub=3, random_seed=42)

k_shot = 5
num_epochs = 1000

src_net = EEGNet(F1=4, D=2, F2=8, in_channel=6, dropout=0.5)
src_net.to(device)
src_optimizer = optim.AdamW(src_net.parameters(), lr=1e-3, weight_decay=1e-4) # Define optimizer
src_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(src_optimizer, T_0=int(num_epochs), eta_min=0.) # Define learning rate scheduler
src_criterion = nn.CrossEntropyLoss() # Define loss

inner_net = EEGNet(F1=4, D=2, F2=8, in_channel=6, dropout=0.5)
inner_net.to(device)
inner_optimizer = optim.SGD(inner_net.parameters(), lr=1e-3)
inner_criterion = nn.CrossEntropyLoss()
num_inner_adapt = 15

save_path = f"{date.today()}_my_few_shot_net.pth"

maml = MAML(
    src_model=src_net, inner_model=inner_net, 
    # train_tasks, val_tasks,
    optimizer=src_optimizer, criterion=src_criterion, scheduler=src_scheduler, num_epochs=num_epochs,
    inner_optimizer=inner_optimizer, inner_criterion=inner_criterion, num_inner_adapt=num_inner_adapt, 
    random_seed=42, device=device,
    save_path=save_path
)
# maml.pre_train(train_tasks, k_shot=k_shot)
# print('model pre_train done...')

for i, (X, y) in enumerate(train_tasks):
    print(f'model fit on train_task{i+1}')
    supt_X, qury_X, supt_y, qury_y = support_query_split(X, y, k_shot=k_shot) # split task to support set and query set.
    maml.fit(supt_X, supt_y, fit_epoch=300)
    print('model fit done...')
    y_hat = maml.predict(qury_X)
    y_predict = np.argmax(y_hat, axis=1)
    correct = sum(y_predict==y)
    acc = correct / len(y)
    print(f'train_tasl_{i+1}: acc = {acc} ({len(y)})')

for i, (X, y) in enumerate(test_tasks):
    print(f'model fit on test_task{i+1}')
    supt_X, qury_X, supt_y, qury_y = support_query_split(X, y, k_shot=k_shot) # split task to support set and query set.
    maml.fit(supt_X, supt_y, fit_epoch=300)
    print('model fit done...')
    y_hat = maml.predict(qury_X)
    y_predict = np.argmax(y_hat, axis=1)
    correct = sum(y_predict==y)
    acc = correct / len(y)
    print(f'test_tasl_{i+1}: acc = {acc} ({len(y)})')