import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='odtrack', choices=['odtrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline_384_ep300', help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate_symtrack(model, template, search, template_tokenfd, search_tokenfd):
    '''Speed Test'''
    macs1, params1 = profile(model, inputs=(template, search, template_tokenfd, search_tokenfd),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template, search, template_tokenfd, search_tokenfd)
        start = time.time()
        for i in range(T_t):
            _ = model(template, search, template_tokenfd, search_tokenfd)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))
        
def evaluate_vit(model, template, search):
    '''Speed Test'''
    macs1, params1 = profile(model, inputs=(template, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template, search)
        start = time.time()
        for i in range(T_t):
            _ = model(template, search)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))
        
def evaluate_vit_separate(model, template, search):
    '''Speed Test'''
    T_w = 50
    T_t = 1000
    print("testing speed ...")
    z = model.forward_backbone(template, image_type='template')
    x = model.forward_backbone(search, image_type='search')
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    try:
        tokenfd_sz = cfg.DATA.TOKENFD_SIZE 
    except AttributeError:
        sys.exit(1)

    if args.script == "symtrack":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_symtrack
        model = model_constructor(cfg, training=False)
        # get the template and search
        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        template_tokenfd = torch.randn(bs, 3, tokenfd_sz, tokenfd_sz) 
        search_tokenfd = torch.randn(bs, 3, tokenfd_sz, tokenfd_sz)
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        
        template_tokenfd = template_tokenfd.to(device)
        search_tokenfd = search_tokenfd.to(device)

        template_list, search_list = [], [search]
        template_tokenfd_list, search_tokenfd_list = [], [search_tokenfd]
        template_len = 3
        for i in range(template_len):
            template_list.append(template)
            template_tokenfd_list.append(template_tokenfd)
            
        merge_layer = cfg.MODEL.BACKBONE.MERGE_LAYER
        if merge_layer <= 0:
            evaluate_symtrack(model, template_list, search_list, template_tokenfd_list, search_tokenfd_list)
        else:
            evaluate_vit_separate(model, template, search)
            
    else:
        raise NotImplementedError
