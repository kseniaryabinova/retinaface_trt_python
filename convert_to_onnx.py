from __future__ import print_function

import torch

from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


weights_path = 'weights/mobilenet0.25_Final.pth'
network_arch = 'mobile0.25'
input_tensor_shape = (1080, 1920)

torch.set_grad_enabled(False)
cfg = None
if network_arch == "mobile0.25":
    cfg = cfg_mnet
elif network_arch == "resnet50":
    cfg = cfg_re50

is_gpu_avaliable = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu_avaliable else "cpu")

net = RetinaFace(cfg=cfg, phase='test')
net = load_model(net, weights_path)
net.eval()
net = net.to(device)

# ------------------------ export -----------------------------
output_onnx = 'retinaface_{}.onnx'.format(network_arch)
input_names = ["input0"]
output_names = ["output0"]
inputs = torch.randn(1, 3, *input_tensor_shape).to(device)

torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                               input_names=input_names, output_names=output_names)
