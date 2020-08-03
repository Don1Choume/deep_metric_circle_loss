import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

