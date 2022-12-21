from torch import nn


#################################################################################################
# initial weights
#################################################################################################
def weights_init_xavier_normal(m):
    classname = m.__class__.__name__

    if classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def weights_init_orthogonal_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
