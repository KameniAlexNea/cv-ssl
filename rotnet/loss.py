from torch.nn import functional as F

def rotnet_loss_fun(z1, y, feats1, feats2):
    sim_loss = F.mse_loss(feats1, feats2)
    rot_loss = F.cross_entropy(z1, y)
    return sim_loss, rot_loss