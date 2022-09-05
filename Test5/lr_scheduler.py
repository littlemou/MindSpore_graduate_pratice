import numpy as np

def get_lr(total_epochs,
           steps_per_epoch,
           lr_init=0.01,
           lr_max=0.1,
           warmup_epochs=5):
    """
    生成学习率数组
    
    Args:
        total_epochs (int): 总epoch数
        steps_per_epoch (float): 每个epoch多少step
        lr_init (float): 初始学习率
        lr_max (float): 最大学习率
        warmup_epochs (int): 预热epoch数

    Returns:
        numpy.ndarray，学习率数组
    """

    lr_each_step = [] # 学习率数组（回传）
    total_steps = steps_per_epoch * total_epochs # 总step数
    warmup_steps = steps_per_epoch * warmup_epochs # 预热step数

    # 计算预热阶段学习率递增值
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0

    # 学习率调整
    for i in range(int(total_steps)):
        if i < warmup_steps:
        	# 预热（学习率线性递增）
            lr = float(lr_init) + inc_each_step * float(i)
        else:
        	# 衰减（学习率指数递减）
            base = ( 1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)) )
            lr = float(lr_max) * base * base
            if lr < 0.0:
                lr = 0.0

       	# 记录学习率
        lr_each_step.append(lr)

    lr_each_step = np.array(lr_each_step).astype(np.float32)

    return lr_each_step