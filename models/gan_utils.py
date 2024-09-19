'''
Utils for cGAN

Creted: 9/19/2024
Updated: 9/19/2024
'''
####################################################################################################################################
# Libraries
####################################################################################################################################
import torch
import torch.nn as nn
####################################################################################################################################
# Index
####################################################################################################################################

#1 set_requires_grad() -> func
#2 ModelInitializer    -> class
#3 

####################################################################################################################################
# Functions
####################################################################################################################################

####################################################################################################################################
# 1

def set_requires_grad(model, requires_grad=True):
    for p in model.parameters():
        p.requires_grad = requires_grad

####################################################################################################################################
# 2

class ModelInitializer:
    def __init__(self, device, init_type='norm', gain=0.02):
        """
        Initializes a model on the specified device with a chosen weight initialization method.
        
        Args:
            device (torch.device): The device to move the model to (e.g., 'cuda' or 'cpu').
            init_type (str): The initialization method ('norm', 'xavier', 'kaiming'). Default is 'norm'.
            gain (float): The gain for initialization. Default is 0.02.
        """
        self.device = device
        self.init_type = init_type
        self.gain = gain
        

    def init_weights(self, net):
        """
        Initializes the weights of the network according to the specified method.
        
        Args:
            net (torch.nn.Module): The model whose weights will be initialized.
            
        Returns:
            torch.nn.Module: The model with initialized weights.
        """
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and 'Conv' in classname:
                if self.init_type == 'norm':
                    nn.init.normal_(m.weight.data, mean=0.0, std=self.gain)
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=self.gain)
                elif self.init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif 'BatchNorm2d' in classname:
                nn.init.normal_(m.weight.data, 1., self.gain)
                nn.init.constant_(m.bias.data, 0.0)

        net.apply(init_func)
        print(f"Model initialized with {self.init_type} initialization")
        return net

    def init_model(self, model, device):
        """
        Returns the initialized model.

        Args:
            model (torch.nn.Module): The model to be initialized.
        Returns:
            torch.nn.Module: The model initialized and moved to the specified device.
        """
        self.model = model.to(self.device)

        # Initialize weights of the model
        self.model = self.init_weights(self.model)

        return self.model


####################################################################################################################################
# 3




####################################################################################################################################
# END
####################################################################################################################################
if __name__ == "__Main__":
    pass