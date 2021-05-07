import torch
import torchvision
from my_model.GFeature_based import Multi_Task
model = Multi_Task()#自己定义的网络模型
model.load_state_dict(torch.load("/home/liqiang/Documents/FOG_DETECTION/model_parameters/rgbd.pth"))#保存的训练模型
model.eval()#切换到eval（）
example1 = torch.rand(1, 4, 448, 448)#生成一个随机输入维度的输入
example2 = torch.rand(1, 3, 448, 448)
traced_script_module = torch.jit.trace(model, example1)
traced_script_module.save("model.pt")
