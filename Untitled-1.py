# %%
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

# 导入自定义库
from models.clamer import *
from utils.utils import *
from datasets.data_loader import *
from utils.plot_figures import *

# %%
cudnn.benchmark = True
cudnn.enabled = True

train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

log_dir = './logs/'
save_best_weight_path = './checkpoints/'

now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

# %%
# 读取数据信息训练集和测试集
train_smiles = read_strings('./data/train_smiles.csv', idx=False)
train_zeo = read_vec('./data/train_zeo.csv', idx=False)
train_syn = read_vec('./data/train_syn.csv', idx=False)
train_codes = read_strings('./data/train_codes.csv', idx=False)
test_smiles = read_strings('./data/test_smiles.csv', idx=False)
test_zeo = read_vec('./data/test_zeo.csv', idx=False)
test_syn = read_vec('./data/test_syn.csv', idx=False)
test_codes = read_strings('./data/test_codes.csv', idx=False)

charset = '?P25$]FO-S.Hc=71(ln63NC4[+)^@'
charlen = len(charset)
print('the charset(inculde begin end and pad char) achieved from dataset :', charset)
print('the total num of charset is :', charlen)
#原始数据编码
char_to_index = dict((c, i) for i, c in enumerate(charset))
index_to_char = dict((i, c) for i, c in enumerate(charset))
char_list = [k for k, v in char_to_index.items()]

# 超参数设置
d_model = 128
head = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 512
epoch = 10
seqlen = 127


src_smiles, tgt_smiles = smiles_padding(train_smiles)
tgt_seq = smiles_to_sequence(tgt_smiles, char_to_index)
tgt_seq = torch.cat([torch.unsqueeze(seq, 0) for seq in tgt_seq]).long()
src_smiles_test, tgt_smiles_test = smiles_padding(test_smiles)
tgt_seq_test = smiles_to_sequence(tgt_smiles_test, char_to_index)
tgt_seq_test = torch.cat([torch.unsqueeze(seq, 0) for seq in tgt_seq_test]).long()
# 划分数据集
train_dataset = SeqDataset(train_zeo, train_syn, tgt_seq)
test_dataset = SeqDataset(test_zeo, test_syn, tgt_seq_test)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 加载模型
model = GptCovd(d_model=d_model, charlen=charlen, device=device, head=head, char_to_index=char_to_index).to(device, non_blocking=True)
# loss
loss_func = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
optim = torch.optim.Adam(model.parameters(), lr=2e-3)
# sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
total = sum(p.numel() for p in model.parameters())
print('total parameters: %0.2fM' % (total / 1e6))  #  打印参数

# %%
# 训练函数
def train_and_test(epochs, criterion, optimizer):
    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
    train_num_samples = len(train_dataset)
    test_num_samples = len(test_dataset)
    # test_num_samples = len(test_dataset)
    train_loss_min = np.inf
    train_acc_max = 0
    test_loss_min = np.inf
    test_acc_max = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        bar = tqdm(total=len(train_dataloader), ncols=125)
        bar.set_description_str(f'{epoch}/{epochs}')

        for i, (zeo, syn, seq) in enumerate(train_dataloader):
            zeo, syn, seq = zeo.to(device, non_blocking=True), syn.to(device, non_blocking=True), seq.to(device,
                                                                                                         non_blocking=True)
            # x = [batch, 24] y = [batch, 126]
            # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字
            pred = model(zeo, syn, seq[:, :-1])[:, 2:, :].to(device, non_blocking=True)  # [8, 50, 39]
            pred = pred.reshape(-1, charlen).to(device, non_blocking=True)  # [8, 50, 39] -> [400, 39]
            y = seq[:, 1:].reshape(-1).to(device, non_blocking=True)  # [8, 50] -> [400]

            select = y != char_to_index['?']  # 忽略pad
            pred = pred[select].to(device, non_blocking=True)
            y = y[select].to(device, non_blocking=True)

            loss = criterion(pred, y).to(device, non_blocking=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss  # if i % 200 == 0:   [select, 28] -> [select]
            pred = pred.argmax(1)
            correct = (pred == y).sum().item()
            accuracy = correct
            train_acc += accuracy
            # lr = optim.param_groups[0]['lr']
            bar.update()

        # 计算训练损失和训练精度
        train_loss_this_epoch = float(train_loss) / (train_num_samples * 125)
        train_acc_this_epoch = float(train_acc) / (train_num_samples * 125)
        train_loss_history.append(train_loss_this_epoch)
        train_acc_history.append(train_acc_this_epoch)

        # 保存最佳模型
        if train_loss_this_epoch < train_loss_min:
            print(
                f"\r\nTrain loss decreased ({train_loss_min:.6f} -> {train_loss_this_epoch:.6f}). Saving model weight...")
            train_loss_min = train_loss_this_epoch
            path = os.path.join(save_best_weight_path,
                                f'NO.{epoch}-{now}-{train_loss_min:.6f}-{train_acc_this_epoch:.2f}.pth')
            torch.save(model.state_dict(), path)
        else:
            train_loss_min = train_loss_min

        if train_acc_this_epoch > train_acc_max:
            print(f"\r\nTrain acc increased ({train_acc_max:.2f} -> {train_acc_this_epoch:.2f})")
            train_acc_max = train_acc_this_epoch
        else:
            train_acc_max = train_acc_max

        bar.set_postfix_str(f'Train Loss:{train_loss_this_epoch:.6f}|Train Acc:{train_acc_this_epoch:.3f}')
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Train_loss', train_loss_history[-1], epoch)
        writer.add_scalar('Train_acc', train_acc_history[-1], epoch)
        # scheduler.step()
        bar.update()
        bar.close()
    
        # 测试
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for i, (zeo, syn, seq) in enumerate(test_dataloader):
                zeo, syn, seq = zeo.to(device, non_blocking=True), syn.to(device, non_blocking=True), seq.to(device,
                                                                                                            non_blocking=True)
                pred = model(zeo, syn, seq[:, :-1])[:, 2:, :].to(device, non_blocking=True)
                pred = pred.reshape(-1, charlen).to(device, non_blocking=True)
                y = seq[:, 1:].reshape(-1).to(device, non_blocking=True)

                select = y != char_to_index['?']
                pred = pred[select].to(device, non_blocking=True)
                y = y[select].to(device, non_blocking=True)

                loss = criterion(pred, y).to(device, non_blocking=True)
                test_loss += loss
                pred = pred.argmax(1)
                correct = (pred == y).sum().item()
                accuracy = correct
                test_acc += accuracy
        # 计算测试损失和测试精度
        test_loss_this_epoch = float(test_loss) / (test_num_samples * 125)
        test_acc_this_epoch = float(test_acc) / (test_num_samples * 125)
        test_loss_history.append(test_loss_this_epoch)
        test_acc_history.append(test_acc_this_epoch)
        # 打印测试损失和测试精度
        print(f'\r\nTest Loss:{test_loss_this_epoch:.6f}|Test Acc:{test_acc_this_epoch:.3f}')

# %%
train_and_test(epochs=epoch, criterion=loss_func, optimizer=optim)  # 训练
plot_loss(train_loss_history, test_loss_history)  # 绘制损失曲线

# %%
# 预测模块
def predict(zeo, syn):
    # x = [1, 50]
    model.eval()

    target = [char_to_index['^']] + [char_to_index['?']] * 124
    target = torch.LongTensor(target).unsqueeze(0)
    # 遍历生成第1个词到第49个词
    for i in range(124):
        # [1, 50]
        out = F.softmax(model(zeo, syn, target)[:, i + 2, :])
        out = out.argmax(dim=1).detach()
        # 以当前词预测下一个词,填到结果中
        target[:, i + 1] = out

    return target


