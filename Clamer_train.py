import pandas as pd
import numpy as np
import math, copy, time
from pandas import DataFrame
import csv
import os
from rdkit import Chem
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import Module
from torch.nn import functional as F
from tqdm import tqdm
from datetime import datetime
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.enabled = True

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

log_dir = './history_241216/'
save_best_weight_path = './history_241216/'
# data_code_file = r'C:\Users\xxx\Desktop\徐留扣-毕业材料\4-源程序\第三章\data_codes'
# data_smiles_file = r'C:\Users\xxx\Desktop\徐留扣-毕业材料\4-源程序\第三章\data_smiles'
# data_synvec_file = r'C:\Users\xxx\Desktop\徐留扣-毕业材料\4-源程序\第三章\data_syn_vectors'
# data_zeovec_file = r'C:\Users\xxx\Desktop\徐留扣-毕业材料\4-源程序\第三章\data_zeo_vectors'

data_code_file = r'./data_codes'
data_smiles_file = r'./data_smiles'
data_synvec_file = r'./data_syn_vectors'
data_zeovec_file = r'./data_zeo_vectors'

train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []
now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def read_vec(filepath, idx=True):  # 定义文件名称，本文件要与当前的.py文件要在同一文件夹下，不然要用绝对路径
    if idx:
        with open(filepath, 'r') as csvfile:  # 打开数据文件
            reader = csv.reader(csvfile)  # 用csv的reader函数读取数据文件
            header = next(reader)  # 读取数据文件的表头
            data = []  # 定义一个空数组用于保存文件的数据
            for line in reader:  # 循环读取数据文件并保存到数组data中
                for i, s in enumerate(line):
                    line[i] = float(s)
                data.append(line)  # line是个一维数组，是数据文件中的一行数据
        return np.array(data)[:, 1:]
    else:
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            data = []
            for line in reader:
                data.append(line)
        return np.array(data)


# 读取生成的smiles字符串csv文件（源数据的whim-pca数据）


def read_strings(filepath, idx=True):
    if idx:
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            data = []
            for line in reader:
                data.append(line)
        return np.array(data)[:, 1:]
    else:
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            data = []
            for line in reader:
                data.append(line)
        return np.array(data)


# 字符串填充
def smiles_padding(smiles):
    src_smiles = []
    tgt_smiles = []
    start_char = '^'
    end_char = '$'
    pad_char = '?'
    for smis in smiles:
        if isinstance(smis, np.ndarray):
            smis = smis[0]
        pad_smile = start_char + smis + end_char + pad_char * 125
        src_smiles.append(pad_smile[0:125])
        tgt_smiles.append(pad_smile[0:126])
    return src_smiles, tgt_smiles

# 数据划分
def data_split(split, unique_codes, smiles, zeo_vectors, syn_vectors, codes):
    train_smiles, train_zeo, train_syn, train_codes = [], [], [], []
    test_smiles, test_zeo, test_syn, test_codes = [], [], [], []
    if split is None:
        train_smiles = smiles
        train_zeo = zeo_vectors
        train_syn = syn_vectors
        train_codes = codes
    elif split == 'random':
        unique_smiles = list(np.unique(smiles))
        print('Number of unique smiles:', len(unique_smiles))
        print(len(unique_smiles))
        test_indices = []
        random.shuffle(unique_smiles)
        test_smiles_unique = unique_smiles[:round(0.2 * len(unique_smiles))]  # 20% held out set
        print(len(test_smiles_unique))
        # for t in test_smiles_unique:
        #     for i, s in enumerate(smiles):
        #         if t == s:
        #             test_indices.append(i)
        for idx, t in enumerate(test_smiles_unique):
            for i, s in enumerate(smiles):
                if t == s:
                    test_indices.append(i)
            print('进度:', idx, '/', len(test_smiles_unique))
        print(len(test_indices))
        for i, (s, z, v, c) in enumerate(zip(smiles, zeo_vectors, syn_vectors, codes)):
            if i in test_indices:
                test_smiles.append(s)
                test_zeo.append(z)
                test_syn.append(v)
                test_codes.append(c)
            else:
                train_smiles.append(s)
                train_zeo.append(z)
                train_syn.append(v)
                train_codes.append(c)
    else:
        if split not in unique_codes:
            print('Problem:', split, 'not a zeolite in the data')
        else:
            for i, (s, z, v, c) in enumerate(zip(smiles, zeo_vectors, syn_vectors, codes)):
                if split == c:
                    test_smiles.append(s)
                    test_zeo.append(z)
                    test_syn.append(v)
                    test_codes.append(c)
                else:
                    train_smiles.append(s)
                    train_zeo.append(z)
                    train_syn.append(v)
                    train_codes.append(c)
    return train_smiles, train_zeo, train_syn, train_codes, test_smiles, test_zeo, test_syn, test_codes

# 字符串编码
def smiles_to_sequence(smis_list):
    smis_list_sequence = []
    for smis in smis_list:
        smis_seq = []
        for c in smis:
            smis_seq.append(char_to_index.get(c))
        smis_seq = torch.tensor(smis_seq, dtype=torch.float32)
        smis_list_sequence.append(smis_seq)
    return smis_list_sequence

# 注意力机制
def attention(Q, K, V, mask):
    # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q,K,V = [b, 4, 50, 8]

    # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
    # Q,K矩阵相乘,求每个词相对其他所有词的注意力
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # 除以每个头维数的平方根,做数值缩放
    score /= 64 ** 0.5

    # mask遮盖,mask是true的地方都被替换成-inf,这样在计算softmax的时候,-inf会被压缩到0
    # mask = [b, 1, 50, 50]
    score = score.masked_fill_(mask, -float('inf'))
    score = torch.softmax(score, dim=-1)

    # 以注意力分数乘以V,得到最终的注意力结果
    # [b, 4, 50, 50] * [b, 4, 50, 8] -> [b, 4, 50, 8]
    score = torch.matmul(score, V)

    # 每个头计算的结果合一
    # [b, 4, 50, 8] -> [b, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 127, d_model)

    return score

# 多头注意力计算层
class MultiHead_Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(d_model, d_model)
        self.fc_K = torch.nn.Linear(d_model, d_model)
        self.fc_V = torch.nn.Linear(d_model, d_model)

        self.out_fc = torch.nn.Linear(d_model, d_model)

        # 规范化之后,均值是0,标准差是1
        # BN是取不同样本做归一化
        # LN是取不同通道做归一化
        # affine=True,elementwise_affine=True,指定规范化后,再计算一个线性映射
        # norm = torch.nn.BatchNorm1d(num_features=4, affine=True)
        # print(norm(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)))

        self.norm = torch.nn.LayerNorm(normalized_shape=d_model, elementwise_affine=True)

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, Q, K, V, mask):
        # b句话,每句话50个词,每个词编码成32维向量
        # Q,K,V = [b, 50, 32]
        # b = Q.shape[0]

        # 保留下原始的Q,后面要做短接用
        clone_Q = Q.clone()

        # 规范化
        Q = self.norm(Q.to(torch.float32))
        K = self.norm(K.to(torch.float32))
        V = self.norm(V.to(torch.float32))

        # 线性运算,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        # 拆分成多个头
        # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 50, 32] -> [b, 4, 50, 8]
        Q = Q.reshape(-1, 127, head, int(d_model / head)).permute(0, 2, 1, 3)
        K = K.reshape(-1, 127, head, int(d_model / head)).permute(0, 2, 1, 3)
        V = V.reshape(-1, 127, head, int(d_model / head)).permute(0, 2, 1, 3)

        # 计算注意力
        # [b, 4, 50, 8] -> [b, 50, 32]
        score = attention(Q, K, V, mask)

        # 计算输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.dropout(self.out_fc(score))

        # 短接
        score = clone_Q + score
        return score

# 嵌入编码
class EmbeddingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # pos是第几个词,i是第几个维度,d_model是维度总数
        def get_pe(pos, i, d_model):
            fenmu = 1e4 ** (i / d_model)
            pe = pos / fenmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        # 初始化位置编码矩阵
        pe = torch.empty(125, d_model)
        for i in range(125):
            for j in range(d_model):
                pe[i, j] = get_pe(i, j, d_model)
        pe = pe.unsqueeze(0)

        # 定义为不更新的常量
        self.register_buffer('pe', pe)

        # 词编码层
        self.char_embed = torch.nn.Embedding(charlen, d_model)
        self.type_embed = torch.nn.Embedding(3, d_model)
        # 初始化参数
        self.char_embed.weight.data.normal_(0, 0.1)
        self.type_embed.weight.data.normal_(0, 0.1)

    def forward(self, zeo, syn, smis_seq):  # zeo, syn,

        b, t = smis_seq.size()
        zeo_te = self.type_embed(torch.zeros((b, 1), dtype=torch.long).to(device, non_blocking=True))
        syn_te = self.type_embed(torch.ones((b, 1), dtype=torch.long).to(device, non_blocking=True))
        smis_seq_te = self.type_embed(torch.ones((b, t), dtype=torch.long, device=device) * 2)

        smis_seq_ce = self.char_embed(smis_seq)  # .to(device)

        # 词编码\位置编码\类型编码相加
        smis_seq_embed = smis_seq_ce.to(device, non_blocking=True) + self.pe.to(device,
                                                                                non_blocking=True) + smis_seq_te.to(
            device, non_blocking=True)
        zeo_embed = zeo.to(device, non_blocking=True) + zeo_te.to(device, non_blocking=True)
        syn_embed = syn.to(device, non_blocking=True) + syn_te.to(device, non_blocking=True)
        return smis_seq_embed.to(device, non_blocking=True), zeo_embed.to(device, non_blocking=True), syn_embed.to(
            device, non_blocking=True)

# 全连接输出层
class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model, out_features=d_model * 4),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=d_model * 4, out_features=d_model),
            torch.nn.Dropout(p=0.5),
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=d_model,
                                       elementwise_affine=True)

    def forward(self, x):
        # 保留下原始的x,后面要做短接用
        clone_x = x.clone()

        # 规范化
        x = self.norm(x)

        # 线性全连接运算
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(x)

        # 做短接
        out = clone_x + out

        return out

# 数据集预制
class SeqDataset(Dataset):
    def __init__(self, zeo, syn, smis_seq):
        assert len(zeo) == len(syn) == len(smis_seq)
        # convert to tensor
        if not torch.is_tensor(zeo):
            # 先转化成浮点数,再转化成张量
            float_zeo = zeo.astype(np.float32)
            self.zeo = torch.tensor(float_zeo, dtype=torch.float32)
        else:
            self.zeo = zeo
            
        if not torch.is_tensor(syn):
            float_syn = syn.astype(np.float32)
            self.syn = torch.tensor(float_syn, dtype=torch.float32)
        else:
            self.syn = syn
        self.smis_seq = smis_seq

    def __getitem__(self, index):  # get sample pair
        return self.zeo[index], self.syn[index], self.smis_seq[index]

    def __len__(self):  # get sample_num
        return len(self.zeo)

# 注意力掩码
def mask_tril(data):
    # b句话,每句话50个词,这里是还没embed的
    # data = [b, 127]
    # 矩阵表示每个词对其他词是否可见
    # 上三角矩阵,不包括对角线,意味着,对每个词而言,他只能看到他自己,和他之前的词,而看不到之后的词
    # [1, 127, 127]
    tril = 1 - torch.tril(torch.ones(1, 127, 127, dtype=torch.long))
    tril = tril.to(device, non_blocking=True)
    # 判断y当中每个词是不是pad,如果是pad则不可见
    # [b, 50]
    b = data.shape[0]
    seq = torch.tensor([1, 2]).reshape(1, -1).expand(b, 2).to(device, non_blocking=True)
    seq = torch.cat((seq, data), dim=-1)
    mask = seq == char_to_index['?']
    mask = mask.to(device, non_blocking=True)
    # 变形+转型,为了之后的计算
    # [b, 1, 50]
    mask = mask.unsqueeze(1).long()
    # mask和tril求并集
    # [b, 1, 50] + [1, 50, 50] -> [b, 50, 50]
    mask = mask + tril
    # 转布尔型
    mask = mask > 0
    # 转布尔型,增加一个维度,便于后续的计算
    mask = (mask == 1).unsqueeze(dim=1)
    return mask

# 解码网络
class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mh = MultiHead_Attn()
        self.fc = FullyConnectedOutput()

    def forward(self, x, mask):
        # 计算自注意力,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.mh(x, x, x, mask)
        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(score)

        return out

# 解码器
class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()
        self.layer_4 = DecoderLayer()
        self.layer_5 = DecoderLayer()
        self.layer_6 = DecoderLayer()

    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        x = self.layer_4(x, mask)
        x = self.layer_5(x, mask)
        x = self.layer_6(x, mask)
        return x

# lstm层
class LSTMLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)

        return x

# gpt架构模块
class GptCovd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_zeo = torch.nn.Linear(7, d_model)
        self.fc_syn = torch.nn.Linear(17, d_model)
        self.embed = EmbeddingLayer()
        self.decoder = Decoder()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.fc_out = torch.nn.Linear(d_model, charlen)
        self.lm = LSTMLayer()

    def forward(self, zeo, syn, smis_seq):
        mask = mask_tril(smis_seq).to(device, non_blocking=True)
        zeo = self.fc_zeo(zeo.to(torch.float32)).reshape((-1, 1, d_model)).to(device, non_blocking=True)
        syn = self.fc_syn(syn.to(torch.float32)).reshape((-1, 1, d_model)).to(device, non_blocking=True)
        # 编码,添加位置信息
        smis_seq_embed, zeo_embed, syn_embed = self.embed(zeo, syn, smis_seq)

        x = torch.cat((zeo_embed, syn_embed, smis_seq_embed), dim=1).to(device, non_blocking=True)

        x = self.lm(x) + x
        x = self.norm1(x)
        # 解码层计算
        y = self.decoder(x, mask)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 39]
        y = self.norm2(y).to(device, non_blocking=True)
        y = self.fc_out(y).to(device, non_blocking=True)

        return y

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

# 绘制训练和测试损失曲线
import matplotlib.pyplot as plt
def plot_loss(train_loss, test_loss):
    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.legend()
    # 保存图片
    plt.savefig(f'./figures/loss.png')
    plt.show()

# 损失函数，没有用到
class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''

    def __init__(self, label_smooth=None, class_num=None):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-2

        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # 转换成one-hot

            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)

        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))

        return loss.sum()


if __name__ == '__main__':
    # 读取数据信息
    # smiles = read_strings(data_smiles_file, idx=True)
    # codes = read_strings(data_code_file, idx=False)
    # 得到不重复的沸石编码
    # unique_codes = np.unique(codes)
    # zeo_features = read_vec(data_zeovec_file, idx=True)
    # syn_features = read_vec(data_synvec_file, idx=True)
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
    epoch = 50
    seqlen = 127
    # 划分训练集和测试集
    # train_smiles, train_zeo, train_syn, train_codes, test_smiles, test_zeo, test_syn, test_codes = data_split(split='random', unique_codes=unique_codes, smiles=smiles, zeo_vectors=zeo_features, syn_vectors=syn_features, codes=codes)
    # 保存划分后的数据集成csv文件
    # pd.DataFrame(train_smiles).to_csv('./data/train_smiles.csv', index=False)
    # pd.DataFrame(train_zeo).to_csv('./data/train_zeo.csv', index=False)
    # pd.DataFrame(train_syn).to_csv('./data/train_syn.csv', index=False)
    # pd.DataFrame(train_codes).to_csv('./data/train_codes.csv', index=False)
    # pd.DataFrame(test_smiles).to_csv('./data/test_smiles.csv', index=False)
    # pd.DataFrame(test_zeo).to_csv('./data/test_zeo.csv', index=False)
    # pd.DataFrame(test_syn).to_csv('./data/test_syn.csv', index=False)
    # pd.DataFrame(test_codes).to_csv('./data/test_codes.csv', index=False)
    
    # 字符填充训练集和测试集
    # src_smiles, tgt_smiles = smiles_padding(smiles)
    # tgt_seq = smiles_to_sequence(tgt_smiles)
    # tgt_seq = torch.cat([torch.unsqueeze(seq, 0) for seq in tgt_seq]).long()
    
    src_smiles, tgt_smiles = smiles_padding(train_smiles)
    tgt_seq = smiles_to_sequence(tgt_smiles)
    tgt_seq = torch.cat([torch.unsqueeze(seq, 0) for seq in tgt_seq]).long()
    src_smiles_test, tgt_smiles_test = smiles_padding(test_smiles)
    tgt_seq_test = smiles_to_sequence(tgt_smiles_test)
    tgt_seq_test = torch.cat([torch.unsqueeze(seq, 0) for seq in tgt_seq_test]).long()
    # 划分数据集
    train_dataset = SeqDataset(train_zeo, train_syn, tgt_seq)
    test_dataset = SeqDataset(test_zeo, test_syn, tgt_seq_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 加载模型
    model = GptCovd().to(device, non_blocking=True)
    # loss
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    # sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
    total = sum(p.numel() for p in model.parameters())
    print('total parameters: %0.2fM' % (total / 1e6))  #  打印参数
    train_and_test(epochs=epoch, criterion=loss_func, optimizer=optim)  # 训练
    plot_loss(train_loss_history, test_loss_history)  # 绘制损失曲线