"""convert lmimo vits backbone checkpoint to detectron2 format"""
import torch
import os
import argparse

def linear2cnn(linear_weight):
    # modify the patch embed: linear to 2D CNN
    return torch.einsum('lpqc->lcpq', linear_weight.reshape(-1,16,16,3))

def cnn2linear(cnn_weight):
    return torch.einsum('lcpq->lpqc', cnn_weight).reshape(-1,16*16*3)

def load_checkpoint(file_path):
    """
    Load a checkpoint from a given file path.

    Parameters:
    - file_path (str): Path to the checkpoint file.

    Returns:
    - state_dict (OrderedDict): The loaded state dictionary.
    """
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    return state_dict

def update_param_names(state_dict):
    """
    Update the names of parameters in the state dictionary.

    Parameters:
    - state_dict (OrderedDict): The model's state dictionary.

    Returns:
    - updated_state_dict (OrderedDict): The updated state dictionary.
    """
    updated_state_dict = state_dict.copy()
    module_in_ckpt = "module" in list(updated_state_dict.keys())[0]
    for k in list(updated_state_dict.keys()):
        if 'encoder' in k and ("ema" not in k) and ("target" not in k):
            if module_in_ckpt:
                prefix = "module.encoder."
            else:
                prefix = "encoder."

            #if checkpoints_pe_mode=="input-pe":
            if "head" not in k:
                updated_state_dict[k[len(prefix):]] = updated_state_dict[k]
        
        # delete renamed or unused k
        del updated_state_dict[k]
    
    updated_state_dict['pos_embed'] = updated_state_dict['pos_embed.pos_embeds'][None]
    updated_state_dict['patch_embed.weight'] = linear2cnn(updated_state_dict['embed.proj.weight'])
    updated_state_dict['patch_embed.bias'] = updated_state_dict['embed.proj.bias']
    del updated_state_dict['embed.proj.weight']
    del updated_state_dict['embed.proj.bias']
    del updated_state_dict['pos_embed.pos_embeds']
    return updated_state_dict

def save_checkpoint(state_dict, file_path):
    """
    Save the state dictionary to a file.

    Parameters:
    - state_dict (OrderedDict): The model's state dictionary.
    - file_path (str): Path to the checkpoint file where the state dictionary will be saved.
    """
    torch.save(state_dict, file_path)
    print(f"Checkpoint saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process checkpoint file")
    parser.add_argument('--checkpoint_epoch', default="latest", type=str, help='The checkpoint epoch')
    parser.add_argument('--ckpt_dir', required=True, type=str, help='The directory where checkpoints are stored')
    parser.add_argument('--pretrain_job_name', required=True, type=str, help='The pretrain job name')
    
    args = parser.parse_args()
    
    checkpoint_epoch = args.checkpoint_epoch
    ckpt_dir = args.ckpt_dir
    pretrain_job_name = args.pretrain_job_name
    
    input_checkpoint_path = os.path.join(ckpt_dir, pretrain_job_name, "checkpoints", f"checkpoint_{checkpoint_epoch}.pth")
    output_checkpoint_path = os.path.join(ckpt_dir, pretrain_job_name, "checkpoints", f"checkpoint_{checkpoint_epoch}_detectron2.pth")

    # input_checkpoint_path = os.path.join("/home/yibingwei/model_zoo", "mae_pretrain_vit_base.pth")

    # Load the checkpoint
    ckpt = load_checkpoint(input_checkpoint_path)
    try:
        state_dict = ckpt["model"]
    except:
        state_dict = ckpt["state_dict"]

    # Update parameter names
    updated_state_dict = update_param_names(state_dict)

    # Save the updated checkpoint
    save_checkpoint(updated_state_dict, output_checkpoint_path)
