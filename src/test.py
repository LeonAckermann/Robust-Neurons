import logging
import os
import torch
import argparse
import numpy as np
import random
from config_parser.parser import create_config
from torch.autograd import Variable
from torch.optim import lr_scheduler
#from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import json
import os
from utils import init_all, valid
from output_init import init_output_function
from torch.nn.parallel import DistributedDataParallel as DDP
import find

def memory_allocation():


    # Set environment variable to enable allocation of the entire GPU memory upfront
    torch.backends.cuda.cufft_plan_cache.clear()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.deterministic = False
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(0)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()

    # Set PyTorch memory management options
    torch.backends.cuda.max_allocated_bytes = 0 # disable check for maximum allocated bytes
    torch.backends.cuda.max_reserved_bytes = 47.5e9  # memory limit RTX A6000 --> 47.5Gb

def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    
    #memory_allocation()
    
    print("number gpus", torch.cuda.device_count())
    print("parsing")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list", default="0")
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt_token_id", default=0)
    parser.add_argument("--compute_baseline", default=False)
    parser.add_argument("--compute_predictivity", default=False)
    parser.add_argument("--compute_accuracy", default=False)
    parser.add_argument("--find_skill_neurons", default=False)
    parser.add_argument("--part", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--mean", default=None)
    parser.add_argument("--std", default=None)
    parser.add_argument("--topn", default=1)
    parser.add_argument("--run", default=None)
    parser.add_argument("--data_path", default=1)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--adversarial", default=False)
    parser.add_argument("--official_submission", default=False)
    parser.add_argument("--perturbation_analysis", default=False)
    parser.add_argument("--prompt_seed", default="unknown")
    parser.add_argument("--compute_in_parts", default=False)
    parser.add_argument("--debugging", default=False)
    parser.add_argument("--mode", default="max_mean_230")
    parser.add_argument("--layer_model_mode", default="layer")
    parser.add_argument("--perturbation_type", default="suppression")
    parser.add_argument("--prompt_dataset", default=None)
    parser.add_argument("--adv_nonAdv", default="False")
    args = parser.parse_args()
    configFilePath = args.config
    config = create_config(configFilePath)

    config.set('prompt', 'prompt_token_id', args.prompt_token_id)
    config.set('eval', 'compute_baseline', args.compute_baseline)
    config.set('eval', 'compute_predictivity', args.compute_predictivity)
    config.set("eval", "find_skill_neurons", args.find_skill_neurons)
    config.set("eval", "part", args.part)
    config.set("eval", "mask", args.mask)
    config.set("eval", "mean", args.mean)
    config.set("eval", "std", args.std)
    config.set("eval", "topn", args.topn)
    config.set("eval", "run", args.run)
    config.set("data", "valid_data_path", args.data_path)
    config.set("output", "output_path", args.output_path)
    config.set("data", "adversarial", args.adversarial)
    config.set("model", "official_submission", args.official_submission)
    config.set("eval", "perturbation_analysis", args.perturbation_analysis)
    config.set("eval", "seed", args.seed)
    config.set("eval", "prompt_seed", args.prompt_seed)
    config.set("eval", "compute_in_parts", args.compute_in_parts)
    config.set("eval", "debugging", args.debugging)
    config.set("eval", "compute_accuracy", args.compute_accuracy)
    config.set("eval", "mode", args.mode)
    config.set("eval", "layer_model_mode", args.layer_model_mode)
    config.set("eval", "perturbation_type", args.perturbation_type)
    config.set("eval", "prompt_dataset", args.prompt_dataset)
    config.set("eval", "adv_nonAdv", args.adv_nonAdv)

    #if torch.backends.mps.is_available():
    #    mps_device = torch.device("mps")
    #    x = torch.ones(1, device=mps_device)
    #    print (x)
    #else:
    #    print ("MPS device not found.")

    

    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#        # list all gpus in case we have ore than one
        device_list = args.gpu.split(",")
        print("device list", device_list)
        for a in range(0, len(device_list)):
        #for a in range(int(args.gpu)):
            gpu_list.append(int(a))

        print("gpu list", gpu_list)

    #config.set('distributed', 'local_rank', args.local_rank)
    #config.set("distributed", "use", False)
    print(config)
    print("number gpus", torch.cuda.device_count())
    if config.getboolean('distributed', "use") and len(gpu_list)>1:
        #torch.cuda.set_device(torch.cuda.device_count())
        print("inside if number gpus", torch.cuda.device_count())

        #os.environ['MASTER_ADDR'] = '172.17.0.3' # get this value with hostname -i
        #os.environ['MASTER_PORT'] = '12355'
        #torch.distributed.init_process_group(rank=args.local_rank, 
        #                                     backend=config.get("distributed", "backend"), 
        #                                     world_size=len(gpu_list))
        #config.set('distributed', 'gpu_num', len(gpu_list))
#
    print('gpu_list',gpu_list)
    cuda = torch.cuda.is_available()
    print('cuda available',cuda)
    #logger.info("CUDA available: %s" % str(cuda))
    #set_random_seed(args.seed)
    
    #for section in config.sections():
    #    print(f"[{section}]")
    #    for (key, val) in config.items(section):
    #        print(f"{key} = {val}")
    #    print()
    #print('finished parsing')
    
    #dataset = init_dataset(dataset_name=config.get("data", "valid_dataset_type"), 
    #                       config=config, 
    #                       mode="valid", 
    #                       batch_size=config.getint("eval", "batch_size"), 
    #                       shuffle=False, 
    #                       reader_num=config.getint("eval", "reader_num"), 
    #                       drop_last=False)

    if config.get("eval", "compute_in_parts") == "True" and config.get("eval", "compute_baseline") == "True":
        total_parts = config.getint("eval", "total_parts")
        for current_part in range(1,total_parts):
            config.set("eval", "current_part", current_part)
            dataset, model, mask = init_all(config, gpu_list=gpu_list)
            #model = DDP(model, device_ids=[0,1])
            print('finished init_dataset')
            print(dataset)
            print(config)#,model
            # Assuming `dataloader` is your DataLoader object
            #for batch in dataset:
            #    print(batch)
            #    break

            output_function = init_output_function(config)

            valid(config=config, 
                  gpu_list=gpu_list,
                  epoch=1,
                  model=model,
                  dataset_name=config.get("data", "valid_dataset_type"),
                  dataset=dataset,
                  output_function=output_function,
                  mask=mask)

        find.sum_to_mean(config, sum=True)
        
    else:

        dataset, model, mask = init_all(config, gpu_list=gpu_list)
        #model = DDP(model, device_ids=[0,1])
        print('finished init_dataset')
        print("length dataset", len(dataset))
        print(config,model)
        # Assuming `dataloader` is your DataLoader object
        #for batch in dataset:
        #    print(batch)
        #    break

        output_function = init_output_function(config)

        valid(config=config, 
              gpu_list=gpu_list,
              epoch=1,
              model=model,
              dataset_name=config.get("data", "valid_dataset_type"),
              dataset=dataset,
              output_function=output_function,
              mask=mask)

    

       




   

