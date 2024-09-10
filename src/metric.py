import torch

def cal_perplexity_token(sample):
    pass

def cal_acc_token(sample, target):
    sample, target = sample.view(-1), target.view(-1)
    target_mask = (target != 256) & (target != 257)
    sample, target = sample[target_mask], target[target_mask]
    return (sample == target).float().mean()


if __name__ == "__main__":
    # sample是从0到9的随机数整数
    sample = torch.randint(0, 10, (1, 10))
    target = torch.randint(0, 10, (1, 10))
    acc = cal_acc_token(sample, target)
