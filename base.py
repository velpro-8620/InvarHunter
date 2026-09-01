import os
import time
import copy

import numpy as np
import torch
from torch import nn
import logging

from improved_model import ImprovedMainModel
from sklearn.metrics import ndcg_score

class BaseModel(nn.Module):
    def __init__(self, event_num, metric_num, node_num, device, lr=1e-3, epoches=50, patience=5, result_dir='./', hash_id=None, **kwargs):
        super(BaseModel, self).__init__()
        
        self.epoches = epoches #训练周期数，即模型将在训练数据上进行多少次训练
        self.lr = lr  #学习率，用于控制模型训练的步长
        self.patience = patience # > 0: use early stop 用于控制早停的参数，如果训练损失在 patience 轮内没有显著下降，则停止训练
        self.device = device #cpu,gpu

        self.model_save_dir = os.path.join(result_dir, hash_id)
        self.model = ImprovedMainModel(event_num, metric_num, node_num, device, **kwargs)
        self.model.to(device)
    
    def evaluate(self, test_loader, datatype="Test"):
        self.model.eval()
        hrs, ndcgs = np.zeros(5), np.zeros(5)
        TP, FP, FN ,TN= 0, 0, 0,0
        batch_cnt, epoch_loss = 0, 0.0 
        #-1
        with torch.no_grad():
            for graph, ground_truths in test_loader:
                res = self.model.forward(graph.to(self.device), ground_truths)
                for idx, faulty_nodes in enumerate(res["y_pred"]):
                    # culprit = ground_truths[idx].item()
                    # if culprit == -1:
                    #     if faulty_nodes[0] == -1: TN+=1
                    #     else: FP += 1
                    # else:
                    #     if faulty_nodes[0] == -1: FN+=1
                    #     else:
                    #         TP+=1
                    #         rank = list(faulty_nodes).index(culprit)
                    #         for j in range(5):
                    #             hrs[j] += int(rank <= j)
                    #             ndcgs[j] += ndcg_score([res["y_prob"][idx]], [res["pred_prob"][idx]], k=j+1)
                    culprit = ground_truths[idx].item()
                    if culprit == 0:  # 现在0表示正常（无故障）
                        if faulty_nodes[0] == 0:
                            TN += 1  # 正确预测为正常
                        else:
                            FP += 1  # 错误预测为异常
                    else:
                        if faulty_nodes[0] == 0:
                            FN += 1  # 未检测出异常
                        else:
                            TP += 1  # 成功识别出异常
                            rank = list(faulty_nodes).index(culprit)   # 防止未命中时报错
                            for j in range(5):
                                hrs[j] += int(rank <= j)
                                ndcgs[j] += ndcg_score([res["y_prob"][idx]], [res["pred_prob"][idx]], k=j + 1)

                epoch_loss += res["loss"].item()
                batch_cnt += 1
        
        pos = TP+FN
        eval_results = {
                "F1": TP*2.0/(TP+FP+pos) if (TP+FP+pos)>0 else 0,
                "Rec": TP*1.0/pos if pos > 0 else 0,
                "Pre": TP*1.0/(TP+FP) if (TP+FP) > 0 else 0}
        
        for j in [1, 3, 5]:
            eval_results["HR@"+str(j)] = hrs[j-1]*1.0/pos
            # eval_results["ndcg@"+str(j)] = ndcgs[j-1]*1.0/pos
            
        logging.info("{} -- {}".format(datatype, ", ".join([k+": "+str(f"{v:.4f}") for k, v in eval_results.items()])))

        return eval_results
    
    def fit(self, train_loader, test_loader=None, evaluation_epoch=10):
        best_hr1, coverage, best_state, eval_res = -1, None, None, None # evaluation
        pre_loss, worse_count = float("inf"), 0
        # early break，pre_loss设置为正无穷，这样可以确保第一次迭代时epoch_loss一定小于pre_loss，从而避免在第一次迭代时就触发早期停止的条件。

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #优化器的作用是根据损失函数的梯度更新模型的参数，从而使得模型能够更好地拟合数据
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99)

        #poch表示训练过程中的迭代轮数。在深度学习中，训练过程是一个迭代优化的过程，通常会使用一定数量的epoch来训练模型，
        #每个epoch表示使用全部训练数据集进行一次前向传播和反向传播的过程
        #每个epoch的结束通常会进行一次模型的评估。迭代轮数越多，模型可能会更加准确，但也可能会过拟合训练数据，因此需要在训练过程中对模型进行评估和调整
        for epoch in range(1, self.epoches+1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0 #在每个 epoch 中，模型会将训练数据分成若干个 batch，分别进行训练，batch_cnt 就是记录当前 epoch 已经处理的 batch 数量
            epoch_time_start = time.time()
            """
            通过计算 loss 函数的梯度，找到一个方向，使得调整参数可以使 loss 函数的值降低.然后根据学习率 lr（lr 参数在初始化训练器对象时设置），
            沿着这个方向对参数进行微调。通过不断迭代，参数会逐渐调整到使得 loss 函数值最小的位置，即模型收敛的位置。
            """
            for graph, label in train_loader:
                optimizer.zero_grad() #梯度清零，梯度被用来更新模型的参数，从而最小化损失函数
                loss = self.model.forward(graph.to(self.device), label)['loss'] #执行前向传播，得到模型的预测结果和损失
                loss.backward() #执行反向传播算法，计算梯度
                # if self.debug:
                #     for name, parms in self.model.named_parameters():
                #         if name=='encoder.graph_model.net.weight':
                #             print(name, "--> grad:",parms.grad)
                optimizer.step() #根据损失函数计算出的梯度对模型中的参数进行更新
                epoch_loss += loss.item() #转换为 Python 标量
                batch_cnt += 1
            epoch_time_elapsed = time.time() - epoch_time_start

            epoch_loss = epoch_loss / batch_cnt
            logging.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epoches, epoch_loss, epoch_time_elapsed))

            ####### early break #######
            #当训练误差不再下降时，会记录连续多少次训练误差没有下降，如果连续多于 self.patience 次训练误差没有下降，则停止训练
            if epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break
            else: worse_count = 0
            pre_loss = epoch_loss

            ####### Evaluate test data during training #######
            if (epoch+1) % evaluation_epoch == 0: #是 evaluation_epoch 的倍数，就调用 evaluate 方法对测试集进行评估
                test_results = self.evaluate(test_loader, datatype="Test")
                if test_results["HR@1"] > best_hr1: #
                    best_hr1, eval_res, coverage  = test_results["HR@1"], test_results, epoch
                    best_state = copy.deepcopy(self.model.state_dict()) #将当前的模型参数复制一份并保存

                self.save_model(best_state) #将当前的模型参数保存在 best_state 变量中，一般可将其保存为文件或数据库记录

        return eval_res, coverage
    
    def load_model(self, model_save_file=""): #加载已经保存的模型参数，需要传入模型参数保存的文件名
        self.model.load_state_dict(torch.load(model_save_file, map_location=self.device))
        #torch.load函数是用于将模型加载到内存中的函数，load_state_dict函数是将加载的模型参数加载到模型中的函数

    def save_model(self, state, file=None):
        if file is None: file = os.path.join(self.model_save_dir, "model.ckpt")
        try:
            torch.save(state, file, _use_new_zipfile_serialization=False) #ZipFile 序列化可以将模型参数保存在多个文件中
        except:
            torch.save(state, file)
