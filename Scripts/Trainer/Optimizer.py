import torch

def get_swin_optimizer(model, base_lr = 1e-5, decay_rate=0.8):
    parameters_group = []

    #Patch embedding and early stages
    parameters_group.append({
        "params" : model.patch_embed.parameters(),
        "lr" : base_lr * (decay_rate ** 2) # 4e-6
    })

   #Gathering parameters from stage1, patch_merge, stage2 and pre-head normalization
    core_transformer_params = (
        list(model.stage1_block.parameters()) +
        list(model.patch_merge.parameters()) +
        list(model.stage2_block.parameters()) +
        list(model.norm.parameters())
    )

    parameters_group.append({
        "params" : core_transformer_params,
        "lr" : base_lr*decay_rate
    })
            
    # Final Classification Head
    parameters_group.append({
        "params": model.fc.parameters(), 
        "lr": base_lr # e.g., 1e-5
    })
    
    optimizer = torch.optim.AdamW(parameters_group, weight_decay=0.05)
    return optimizer