from math import cos, pi

def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min, lr_max, warmup=True):
    warmup_epoch = 10 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    elif current_epoch >= max_epoch:
        lr = lr_min
    else:
        lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr