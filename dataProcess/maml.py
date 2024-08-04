import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
from utils import isSame, support_query_split
import matplotlib.pyplot as plt
from datetime import date

class MyDataset(Dataset):
    def __init__(self, data, label=None):
        if label is None:
            self.y = label
        else:
            self.y = torch.LongTensor(label)
        self.x = torch.FloatTensor(data)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)

class MAML():
    def __init__(self,
        src_model, inner_model,
        optimizer, criterion, scheduler=None, num_epochs=50, # outer loop
        inner_optimizer=None, inner_criterion=None, num_inner_adapt=5, inner_scheduler=None, # inner loop
        random_seed=None, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=32, save_path=f"{date.today()}_my_few_shot_net.pth"

    ):
        isSame(src_model, inner_model)
        
        self.src_model = src_model.to(device)
        self.inner_model = inner_model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        if inner_optimizer is None:
            self.inner_optimizer = optim.SGD(self.inner_model.parameters(), lr=1e-3)
        else:
            self.inner_optimizer = inner_optimizer
        if inner_criterion is None:
            self.inner_criterion = criterion
        else:
            self.inner_criterion = inner_criterion
        self.num_inner_adapt = num_inner_adapt
        self.inner_scheduler = inner_scheduler
        self.random_seed = random_seed
        self.device = device
        self.batch_size = batch_size
        self.save_path = save_path
        
        
        self.loadModel(self.save_path)

    def pre_train(self, train_task, k_shot=5, plot_learning_curve=True):

        avg_train_loss_list = []
        train_acc_list = []

        if plot_learning_curve:
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 8))
        
        for epoch in range(self.num_epochs):
            src_state_dict = copy.deepcopy(self.src_model.state_dict())
            updated_src_state_dict = copy.deepcopy(self.src_model.state_dict())

            total_train_loss = 0
            total_train_samples = 0
            total_correct_num = 0

            for (X, y) in train_task:

                supt_X, qury_X, supt_y, qury_y = support_query_split(X, y, k_shot=k_shot) # split task to support set and query set.

                # inner model adapt.
                supt_set = MyDataset(supt_X, supt_y)
                supt_loader = DataLoader(supt_set, batch_size=self.batch_size, shuffle=True)
                self.inner_model.load_state_dict(src_state_dict) # init inner model state dict.
                self.inner_model.train()

                for _ in range(self.num_inner_adapt):
                    for x, y in supt_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        inner_loss = self.inner_criterion(self.inner_model(x), y)
                        self.inner_optimizer.zero_grad()
                        inner_loss.backward()
                        self.inner_optimizer.step()
                
                adapted_state_dict = copy.deepcopy(self.inner_model.state_dict())
                # print(src_state_dict['classifier.1.weight']==adapted_state_dict['classifier.1.weight'])

                # update src model.
                self.src_model.train()
                qury_set = MyDataset(qury_X, qury_y)
                qury_loader = DataLoader(qury_set, batch_size=self.batch_size, shuffle=True)

                for x, y in qury_loader:
                    self.src_model.load_state_dict(adapted_state_dict) # init src model state dict
                    
                    batch_size = x.shape[0]
                    if batch_size == 1:
                        continue

                    x, y = x.to(self.device), y.to(self.device)
                    y_hat = self.src_model(x)

                    predict = torch.argmax(y_hat, dim=1)
                    total_correct_num += (predict == y).sum().item()

                    train_loss = self.criterion(y_hat, y)
                    total_train_loss += train_loss.item() * batch_size
                    total_train_samples += batch_size

                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.src_model.load_state_dict(updated_src_state_dict)
                    self.optimizer.step()
                    updated_src_state_dict = copy.deepcopy(self.src_model.state_dict())
                
            avg_train_loss_list.append(total_train_loss / total_train_samples)
            train_acc_list.append(total_correct_num / total_train_samples)

            if plot_learning_curve:
                [axs[i].clear() for i in range(len(axs))]
                axs[0].plot(avg_train_loss_list, label='Train Loss', color='r')
                axs[0].set_title('Learning Curve')
                axs[1].plot(train_acc_list, label='Train Acc', color='b')
                axs[1].set_title('Acc Curve')
                plt.tight_layout()
                plt.pause(0.5)

            print(f'epoch_{epoch}: avg_train_loss = {avg_train_loss_list[-1]}, total_train_acc = {train_acc_list[-1]}')
            if self.scheduler is not None:
                self.scheduler.step()

        torch.save(self.src_model.state_dict(), self.save_path)

    def fit(self, X, y, fit_epoch=100):

        supt_set = MyDataset(X, y)
        supt_loader = DataLoader(supt_set, batch_size=self.batch_size, shuffle=True)

        self.loadModel(self.save_path)
        self.src_model.train()

        for epoch in range(fit_epoch):
            for x, y in supt_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.criterion(self.src_model(x), y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
        return self
    def score(self, X, y):
        """
        使用已训练的模型对输入数据X进行评分。
        
        如果未指定模型状态字典的路径，则使用类实例中保存的路径。
        从给定路径加载模型状态字典，并将模型设置为评估模式。
        遍历数据加载器中的批次数据，进行评分并累积结果。
        最后，返回所有输入数据的平均得分。
        
        参数:
        X: 输入数据，用于评分。
        ```
        y: 输入数据的标签。
        state_dict_path: 模型状态字典的路径。如果未指定，则使用类实例中保存的路径。
        
        返回:
        float: 所有输入数据的平均得分。
        """
        yhat = self.predict(X)
        return np.round(np.mean(yhat == y),2) 
            
            
    def loadModel(self, state_dict_path=None):
        if state_dict_path is None:
            return

        src_state_dict = torch.load(state_dict_path)
        self.src_model.load_state_dict(src_state_dict)

    def predict(self, X):
        """
        使用已训练的模型对输入数据X进行预测。
        
        如果未指定模型状态字典的路径，则使用类实例中保存的路径。
        从给定路径加载模型状态字典，并将模型设置为评估模式。
        遍历数据加载器中的批次数据，进行预测并累积结果。
        最后，将所有预测结果垂直堆叠并返回。
        
        参数:
        X: 输入数据，用于预测。
        
        返回:
        np.array: 所有输入数据的预测结果。trail channel time
        """

        
        self.src_model.eval()
        y_hat = []
        set = MyDataset(X)
        loader = DataLoader(set, batch_size=self.batch_size, shuffle=False)
        for x in loader:
            x = x.to(self.device)
            y_hat.append(np.argmax(self.src_model(x).cpu().detach().numpy(), axis = 1)+1)
        return np.vstack(y_hat)
    
        
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from load_Base_de_Datos_Habla_Imaginada import get_all_sub_data_list, down_sample_data, gen_task_list, maml_task_list
    data = get_all_sub_data_list(cls_list=[0,1,2,3,4])
    data = down_sample_data(data)
    task_list = gen_task_list(data)
    train_tasks, test_tasks = maml_task_list(task_list, num_test_sub=3, random_seed=42)

    from eegnet import EEGNet

    k_shot = 3
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
    num_inner_adapt = 5

    # save_path = f"{date.today()}_my_few_shot_net.pth"
    
    save_path = '2024-07-26_my_few_shot_net.pth'

    maml = MAML(
        src_model=src_net, inner_model=inner_net, 
        # train_tasks, val_tasks,
        optimizer=src_optimizer, criterion=src_criterion, scheduler=src_scheduler, num_epochs=num_epochs,
        inner_optimizer=inner_optimizer, inner_criterion=inner_criterion, num_inner_adapt=num_inner_adapt, 
        random_seed=42, device=device,
        save_path=save_path
    )
    # maml.pre_train(train_tasks, k_shot=k_shot)
    print('model pre_train done...')

    for i, (X, y) in enumerate(test_tasks):
        print(f'model fit on test_task{i+1}')
        supt_X, qury_X, supt_y, qury_y = support_query_split(X, y, k_shot=k_shot) # split task to support set and query set.
        maml.fit(supt_X, supt_y)
        print('model fit done...')
        y_hat = maml.predict(qury_X)
        y_predict = np.argmax(y_hat, axis=1)
        correct = sum(y_predict==y)
        acc = correct / len(y)
        print(f'test_tasl_{i+1}: acc = {acc}')
