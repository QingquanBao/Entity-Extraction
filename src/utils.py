from transformers import AdamW

def AdamW_grouped_LLRD(model,init_lr):
    ## ref: https://zhuanlan.zhihu.com/p/412889866
    ## Base in the source ref is roberta
    
    opt_parameters = []       # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 

    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]
    # init_lr = 1e-6

    for i, (name, params) in enumerate(named_parameters):  
        print(name)
        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01

        if name.startswith("bert.embeddings") or name.startswith("bert.encoder"):            
            # For first set, set lr to 1e-6 (i.e. 0.000001)
            lr = init_lr       

            # For set_2, increase lr to 0.00000175
            lr = init_lr * 1.75 if any(p in name for p in set_2) else lr

            # For set_3, increase lr to 0.0000035 
            lr = init_lr * 3.5 if any(p in name for p in set_3) else lr

            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})  

        # For regressor and pooler, set lr to 0.0000036 (slightly higher than the top layer).                
        if name.startswith("regressor") or name.startswith("bert.pooler"):               
            lr = init_lr * 3.6 

            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})    

    return AdamW(opt_parameters, lr=init_lr)

