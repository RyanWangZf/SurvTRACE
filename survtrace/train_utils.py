'''training utils for trainign survtrace model.
'''
from collections import defaultdict
import pdb
import os
import math
import numpy as np
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from torch import optim

from .losses import NLLPCHazardLoss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), name)
        self.val_loss_min = val_loss

def pad_col(input, val=0, where='end'):
    """Addes a column of `val` at the start of end of `input`."""
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

    
############################
# optimizer #
############################

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + math.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}

class BERTAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(BERTAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        print("l_total=",len(self.param_groups))
        for group in self.param_groups:
            print("l_p=",len(group['params']))
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                pdb.set_trace()
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = initial_step
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


############################
# trainer #
############################

class Trainer:
    def __init__(self, model, metrics=None):
        '''metrics must start from NLLPCHazardLoss, then be others
        '''
        self.model = model
        if metrics is None:
            self.metrics = [NLLPCHazardLoss(),]

        self.train_logs = defaultdict(list)
        self.get_target = lambda df: (df['duration'].values, df['event'].values)
        self.use_gpu = True if torch.cuda.is_available() else False
        if self.use_gpu:
            print('use pytorch-cuda for training.')
            self.model.cuda()
            self.model.use_gpu = True
        else:
            print('GPU not found! will use cpu for training!')
        self.early_stopping = None
        ckpt_dir = os.path.dirname(model.config['checkpoint'])
        self.ckpt = model.config['checkpoint']
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    def train_single_event(self,
        train_set,
        val_set=None,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=0,
        val_batch_size=None,
        **kwargs,
        ):

        df_train, df_y_train = train_set
        durations_train, events_train = self.get_target(df_y_train)

        if val_set is not None:
            df_val, df_y_val = val_set
            durations_val, events_val = self.get_target(df_y_val)
            tensor_val = torch.tensor(val_set[0].values)
            tensor_y_val = torch.tensor(val_set[1].values)
        
        if self.use_gpu:
            tensor_val = tensor_val.cuda()
            tensor_y_val = tensor_y_val.cuda()

        # assign no weight decay on these parameters
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BERTAdam(optimizer_grouped_parameters, 
            learning_rate, 
            weight_decay_rate=weight_decay, 
            )

        if val_set is not None:
            # take early stopping
            self.early_stopping = EarlyStopping(patience=self.model.config['early_stop_patience'])

        num_train_batch = int(np.ceil(len(df_y_train) / batch_size))
        train_loss_list, val_loss_list = [], []
        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()
            df_train = train_set[0].sample(frac=1)
            df_y_train = train_set[1].loc[df_train.index]

            tensor_train = torch.tensor(df_train.values)
            tensor_y_train = torch.tensor(df_y_train.values)
            if self.use_gpu:
                tensor_y_train = tensor_y_train.cuda()
                tensor_train = tensor_train.cuda()

            for batch_idx in range(num_train_batch):
                optimizer.zero_grad()

                batch_train = tensor_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                batch_y_train = tensor_y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                
                batch_x_cat = batch_train[:, :self.model.config.num_categorical_feature].long()
                batch_x_num = batch_train[:, self.model.config.num_categorical_feature:].float()

                phi = self.model(input_ids=batch_x_cat, input_nums=batch_x_num)

                if len(self.metrics) == 1: # only NLLPCHazardLoss is asigned
                    batch_loss = self.metrics[0](phi[1], batch_y_train[:,0].long(), batch_y_train[:,1].long(), batch_y_train[:,2].float(), reduction="mean")

                else:
                    raise NotImplementedError

                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss.item()

            train_loss_list.append(epoch_loss / (batch_idx+1))

            if val_set is not None:
                self.model.eval()
                with torch.no_grad():
                    phi_val = self.model.predict(tensor_val, val_batch_size)
                
                val_loss = self.metrics[0](phi_val, tensor_y_val[:,0].long(), tensor_y_val[:,1].long(), tensor_y_val[:,2].float())
                print("[Train-{}]: {}".format(epoch, epoch_loss))
                print("[Val-{}]: {}".format(epoch, val_loss.item()))
                val_loss_list.append(val_loss.item())
                self.early_stopping(val_loss.item(), self.model, name=self.ckpt)
                if self.early_stopping.early_stop:
                    print(f"early stops at epoch {epoch+1}")
                    # load best checkpoint
                    self.model.load_state_dict(torch.load(self.ckpt))
                    return train_loss_list, val_loss_list
            else:
                print("[Train-{}]: {}".format(epoch, epoch_loss))

        return train_loss_list, val_loss_list

    def train_multi_event(self,
        train_set,
        val_set=None,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=0,
        val_batch_size=None,
        **kwargs,
        ):

        if val_set is not None:
            tensor_val = torch.tensor(val_set[0].values)
            tensor_y_val = dict()
            for risk in range(self.model.config.num_event):
                tensor_y_val["risk_{}".format(risk)] = torch.tensor(val_set[1][["duration","event_{}".format(risk),"proportion"]].values).cuda()

            if self.use_gpu:
                tensor_val = tensor_val.cuda()
                for key in tensor_y_val.keys():
                    tensor_y_val[key] = tensor_y_val[key].cuda()
            
        # assign no weight decay on these parameters
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BERTAdam(optimizer_grouped_parameters, 
            learning_rate, 
            weight_decay_rate=weight_decay, 
            )

        if val_set is not None:
            # take early stopping
            self.early_stopping = EarlyStopping(patience=self.model.config['early_stop_patience'])

        train_loss_list, val_loss_list = [], []
        num_train_batch = int(np.ceil(len(train_set[0]) / batch_size))
        for epoch in range(epochs):
            df_train = train_set[0].sample(frac=1)
            df_y_train = train_set[1].loc[df_train.index]

            tensor_train = torch.tensor(df_train.values)
            tensor_y_train = {}
            for risk in range(self.model.config.num_event):
                tensor_y_train["risk_{}".format(risk)] = torch.tensor(df_y_train[["duration","event_{}".format(risk),"proportion"]].values)

            if self.use_gpu:
                tensor_train = tensor_train.cuda()
                for key in tensor_y_train.keys():
                    tensor_y_train[key] = tensor_y_train[key].cuda()
            
            epoch_loss = 0
            for batch_idx in range(num_train_batch):
                optimizer.zero_grad()

                batch_train = tensor_train[batch_idx*batch_size:(batch_idx+1)*batch_size]

                batch_x_cat = batch_train[:, :self.model.config.num_categorical_feature].long()
                batch_x_num = batch_train[:, self.model.config.num_categorical_feature:].float()

                batch_loss = None
                for risk in range(self.model.config.num_event):
                    phi = self.model(input_ids=batch_x_cat, input_nums=batch_x_num, event=risk)
                    batch_y_train = tensor_y_train["risk_{}".format(risk)][batch_idx*batch_size:(batch_idx+1)*batch_size]
                    if len(self.metrics) == 1: # only NLLPCHazardLoss is asigned
                        if batch_loss is None:
                            batch_loss = self.metrics[0](phi[1], batch_y_train[:,0].long(), batch_y_train[:,1].long(), batch_y_train[:,2].float())
                        else:
                            batch_loss += self.metrics[0](phi[1], batch_y_train[:,0].long(), batch_y_train[:,1].long(), batch_y_train[:,2].float())
                    else:
                        raise NotImplementedError

                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

            train_loss_list.append(epoch_loss / (batch_idx+1))
            if val_set is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for risk in range(self.model.config.num_event):
                        phi_val = self.model.predict(tensor_val, val_batch_size, event=risk)
                        val_loss += self.metrics[0](phi_val, tensor_y_val["risk_{}".format(risk)][:,0].long(), tensor_y_val["risk_{}".format(risk)][:,1].long(), tensor_y_val["risk_{}".format(risk)][:,2].float())

                print("[Train-{}]: {}".format(epoch, epoch_loss / (batch_idx+1)))
                print("[Val-{}]: {}".format(epoch, val_loss.item()))
                val_loss_list.append(val_loss.item())
                self.early_stopping(val_loss.item(), self.model, name=self.ckpt)
                if self.early_stopping.early_stop:
                    print(f"early stops at epoch {epoch+1}")
                    # load best checkpoint
                    self.model.load_state_dict(torch.load(self.ckpt))
                    return train_loss_list, val_loss_list
            else:
                print("[Train-{}]: {}".format(epoch, epoch_loss))

        return train_loss_list, val_loss_list

    def fit(self, 
        train_set,
        val_set=None,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=0,
        val_batch_size=None,
        **kwargs,
        ):
        '''fit on the train_set, validate on val_set for early stop
        params should have the following terms:
        batch_size,
        epochs,
        optimizer,
        metric,
        '''
        if self.model.config.num_event == 1:
            return self.train_single_event(
                    train_set=train_set,
                    val_set=val_set,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    val_batch_size=val_batch_size,
                    **kwargs,
            )
        
        elif self.model.config.num_event > 1:
            return self.train_multi_event(
                    train_set=train_set,
                    val_set=val_set,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    val_batch_size=val_batch_size,
                    **kwargs,
            )
        
        else:
            raise ValueError
