# -*- coding: utf-8 -*-


class Config(object):
    debug_file = 'debug_file'
    
    train_data_root = '/train'
    val_data_root = '/test'
    
    output_sample = 'output_sample'
    dehazing_result = 'dehazing_result'
    load_model_path = 'checkpoints/DHID_best.pth'#None

    train_num = 14990

    
    batch_size = 10
    val_batch_size = 1
    num_workers = 4
    
    lr = 0.00005# initial learning rate

    new_lr = 0.00005

    weight_decay = 0.0001 

    
    max_epoch = 500
    display_iter = 100
    sample_iter = 100
    result_sample_iter = 10
    
    lr_decay = 0.70
    

