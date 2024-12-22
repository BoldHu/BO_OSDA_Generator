import math
import torch
import torch.nn as nn

# 超参数设置
class config():
    def __init__(self):
        self.batch_size = 64
        self.d_model = 128
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.charset = '?P25$]FO-S.Hc=71(ln63NC4[+)^@'
        self.charlen = len(self.charset)
        self.char_to_index = dict((c, i) for i, c in enumerate(self.charset))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.charset))
        self.head = 4
        self.seqlen = 127


def attention(Q, K, V, mask):
    # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q,K,V = [b, 4, 50, 8]

    # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
    # Q,K矩阵相乘,求每个词相对其他所有词的注意力
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # 除以每个头维数的平方根,做数值缩放
    score /= int(config().d_model/config().head) ** 0.5

    # mask遮盖,mask是true的地方都被替换成-inf,这样在计算softmax的时候,-inf会被压缩到0
    # mask = [b, 1, 50, 50]
    score = score.masked_fill_(mask, -float('inf'))
    score = torch.softmax(score, dim=-1)

    # 以注意力分数乘以V,得到最终的注意力结果
    # [b, 4, 50, 50] * [b, 4, 50, 8] -> [b, 4, 50, 8]
    score = torch.matmul(score, V)

    # 每个头计算的结果合一
    # [b, 4, 50, 8] -> [b, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 127, config().d_model)

    return score


# 多头注意力计算层
class MultiHead_Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(config().d_model, config().d_model)
        self.fc_K = torch.nn.Linear(config().d_model, config().d_model)
        self.fc_V = torch.nn.Linear(config().d_model, config().d_model)

        self.out_fc = torch.nn.Linear(config().d_model, config().d_model)

        # 规范化之后,均值是0,标准差是1
        # BN是取不同样本做归一化
        # LN是取不同通道做归一化
        # affine=True,elementwise_affine=True,指定规范化后,再计算一个线性映射
        # norm = torch.nn.BatchNorm1d(num_features=4, affine=True)
        # print(norm(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)))

        self.norm = torch.nn.LayerNorm(normalized_shape=config().d_model, elementwise_affine=True)

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
        Q = Q.reshape(-1, 127, config().head, int(config().d_model / config().head)).permute(0, 2, 1, 3)
        K = K.reshape(-1, 127, config().head, int(config().d_model / config().head)).permute(0, 2, 1, 3)
        V = V.reshape(-1, 127, config().head, int(config().d_model / config().head)).permute(0, 2, 1, 3)

        # 计算注意力
        # [b, 4, 50, 8] -> [b, 50, 32]
        score = attention(Q, K, V, mask)

        # 计算输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        score = self.dropout(self.out_fc(score))

        # 短接
        score = clone_Q + score
        return score


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
        pe = torch.empty(125, config().d_model)
        for i in range(125):
            for j in range(config().d_model):
                pe[i, j] = get_pe(i, j, config().d_model)
        pe = pe.unsqueeze(0)

        # 定义为不更新的常量
        self.register_buffer('pe', pe)

        # 词编码层
        self.char_embed = torch.nn.Embedding(config().charlen, config().d_model)
        self.type_embed = torch.nn.Embedding(3, config().d_model)
        # 初始化参数
        self.char_embed.weight.data.normal_(0, 0.1)
        self.type_embed.weight.data.normal_(0, 0.1)

    def forward(self, zeo, syn, smis_seq):  # zeo, syn,

        b, t = smis_seq.size()
        zeo_te = self.type_embed(torch.zeros((b, 1), dtype=torch.long).to(config().device, non_blocking=True))
        syn_te = self.type_embed(torch.ones((b, 1), dtype=torch.long).to(config().device, non_blocking=True))
        smis_seq_te = self.type_embed(torch.ones((b, t), dtype=torch.long, device=config().device) * 2)

        smis_seq_ce = self.char_embed(smis_seq)  # .to(device)

        # 词编码\位置编码\类型编码相加
        smis_seq_embed = smis_seq_ce.to(config().device, non_blocking=True) + self.pe.to(config().device,
                                                                                non_blocking=True) + smis_seq_te.to(
            config().device, non_blocking=True)
        zeo_embed = zeo.to(config().device, non_blocking=True) + zeo_te.to(config().device, non_blocking=True)
        syn_embed = syn.to(config().device, non_blocking=True) + syn_te.to(config().device, non_blocking=True)
        return smis_seq_embed.to(config().device, non_blocking=True), zeo_embed.to(config().device, non_blocking=True),\
               syn_embed.to(config().device, non_blocking=True)


# 全连接输出层
class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=config().d_model, out_features=config().d_model * 4),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=config().d_model * 4, out_features=config().d_model),
            torch.nn.Dropout(p=0.5),
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=config().d_model,
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


def mask_tril(data):
    # b句话,每句话50个词,这里是还没embed的
    # data = [b, 127]
    # 矩阵表示每个词对其他词是否可见
    # 上三角矩阵,不包括对角线,意味着,对每个词而言,他只能看到他自己,和他之前的词,而看不到之后的词
    # [1, 127, 127]
    tril = 1 - torch.tril(torch.ones(1, 127, 127, dtype=torch.long))
    tril = tril.to(config().device, non_blocking=True)
    # 判断y当中每个词是不是pad,如果是pad则不可见
    # [b, 50]
    b = data.shape[0]
    seq = torch.tensor([1, 2]).reshape(1, -1).expand(b, 2).to(config().device, non_blocking=True)
    seq = torch.cat((seq, data), dim=-1)
    mask = seq == config().char_to_index['?']
    mask = mask.to(config().device, non_blocking=True)
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

# lstm
class LSTMLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=config().d_model, hidden_size=config().d_model, num_layers=1, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)

        return x

# 模型结构
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_zeo = torch.nn.Linear(7, config().d_model)
        self.fc_syn = torch.nn.Linear(17, config().d_model)
        self.embed = EmbeddingLayer()
        self.decoder = Decoder()
        self.norm1 = nn.LayerNorm(config().d_model)
        self.norm2 = torch.nn.LayerNorm(config().d_model)
        self.fc_out = torch.nn.Linear(config().d_model, config().charlen)
        self.lm = LSTMLayer()

    def forward(self, zeo, syn, smis_seq):
        mask = mask_tril(smis_seq).to(config().device, non_blocking=True)
        zeo = self.fc_zeo(zeo.to(torch.float32)).reshape((-1, 1, config().d_model)).to(config().device, non_blocking=True)
        syn = self.fc_syn(syn.to(torch.float32)).reshape((-1, 1, config().d_model)).to(config().device, non_blocking=True)
        # 编码,添加位置信息
        smis_seq_embed, zeo_embed, syn_embed = self.embed(zeo, syn, smis_seq)

        x = torch.cat((zeo_embed, syn_embed, smis_seq_embed), dim=1).to(config().device, non_blocking=True)

        x = self.lm(x) + x
        x = self.norm1(x)
        # 解码层计算
        y = self.decoder(x, mask)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 39]
        y = self.norm2(y).to(config().device, non_blocking=True)
        y = self.fc_out(y).to(config().device, non_blocking=True)

        return y



