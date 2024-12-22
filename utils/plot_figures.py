import matplotlib.pyplot as plt

# 绘制训练和测试损失曲线
def plot_loss(train_loss, test_loss, model_name=None):
    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.legend()
    # 保存图片
    if model_name:
        plt.savefig(f'./figures/{model_name}_loss.png')
    else:
        plt.savefig(f'./figures/loss.png')
    plt.show()