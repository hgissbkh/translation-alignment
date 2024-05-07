import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', type=int, default=1)
    parser.add_argument('-cpt', '--cpus-per-task', type=int, default=8)
    parser.add_argument('-mt', '--max-time', type=str, default='5:00:00')
    parser.add_argument('-tk', '--task', type=str, default='evaluate')
    parser.add_argument('-mn', '--model-name', type=str, default='Llama-2-7b-SFT-MT-Blocksv0.2')
    parser.add_argument('-e', '--num-epochs', type=int, default=3)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5)
    parser.add_argument('-wu', '--warmup-ratio', type=float, default=0)
    parser.add_argument('-mbs', '--micro-batch-size', type=int, default=4)
    parser.add_argument('-gas', '--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('-spe', '--saves-per-epoch', type=int, default=5)
    parser.add_argument('-N', '--num-candidates', type=int, default=20)
    parser.add_argument('-t', '--temperature', type=float, default=0.9)
    parser.add_argument('-p', '--top-p', type=float, default=0.6)
    parser.add_argument('-tdn', '--train-dataset-name', type=str, default='mt-align-study-w-idiom-1203')
    parser.add_argument('-edn', '--eval-dataset-name', type=str, default='mt-align-study-w-idiom-1203')
    args = parser.parse_args()

    if args.task == 'build-preference-data':
        job_name = f'prefdata_{args.model_name}_N{args.num_candidates}_t{args.temperature}_p{args.top_p}'
        command = f"""
echo "==========> Generating greedy hypotheses..."
python generate_hypotheses.py -mn "{args.model_name}"

echo "==========> Sampling candidates..."
python generate_hypotheses.py -N {args.num_candidates} -t {args.temperature} -p {args.top_p} -mn "{args.model_name}"

echo "==========> Building preference dataset..."
python build_preference_data.py -cf "hypotheses_N{args.num_candidates}_t{args.temperature}_p{args.top_p}.txt" -mn "{args.model_name}"
    """
        
    elif args.task == 'run-dpo':
        hyperparameters = f'e{args.num_epochs}_bs{args.micro_batch_size * args.gradient_accumulation_steps}_lr{args.learning_rate}_wu{args.warmup_ratio}'
        job_name = f'dpo_{args.model_name.split("-")[0].lower()}_{hyperparameters}'
        yaml_script = f"""
base_model: ./models/{args.model_name}/unaligned
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
is_llama_derived_model: true

load_in_8bit: false
load_in_4bit: false
strict: false

rl: dpo
dpo_beta: 0.7
chat_template:
datasets:
  - path: data/training/{args.train_dataset_name}/{args.model_name}/preference-data-N{args.num_candidates}-t{args.temperature}-p{args.top_p}
    split: train
    type:
dataset_prepared_path:
val_set_size:
output_dir: ./models/{args.model_name}/{hyperparameters}
wandb_mode: offline

sequence_len: 1024
sample_packing: false
pad_to_sequence_len: false

adapter: 
lora_model_dir:
lora_r:
lora_alpha:
lora_dropout:
lora_target_linear:
lora_fan_in_fan_out:
lora_target_modules:

gradient_accumulation_steps: {args.gradient_accumulation_steps}
micro_batch_size: {args.micro_batch_size}
num_epochs: {args.num_epochs}
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: {args.learning_rate}

train_on_inputs: false
group_by_length: false
bf16: true
fp16: false
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_ratio: {args.warmup_ratio}
evals_per_epoch:
eval_table_size: 
saves_per_epoch: {args.saves_per_epoch}
save_steps:
save_total_limit: {args.saves_per_epoch * args.num_epochs}
max_steps: 
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
"""
                
        with open(f'config/{job_name}.yaml', 'w') as yaml_file:
            yaml_file.write(yaml_script)
        
        command = f"""
accelerate launch -m axolotl.cli.train config/{job_name}.yaml
    """
    
    elif args.task == 'evaluate':
        try:
            hyperparameters = f'e{args.num_epochs}_bs{args.micro_batch_size * args.gradient_accumulation_steps}_lr{args.learning_rate}_wu{args.warmup_ratio}'
            job_name = f'eval_{args.eval_dataset_name.split("-")[0]}_{args.model_name.split("-")[0].lower()}_{hyperparameters}'
            chkpts = [dir_name for dir_name in os.listdir(f'models/{args.model_name}/{hyperparameters}') if dir_name.startswith('checkpoint')]
            chkpts = ['unaligned'] + chkpts + ['aligned']
        except:
            hyperparameters = 'None'
            job_name = f'eval_{args.eval_dataset_name.split("-")[0]}_{args.model_name.split("-")[0].lower()}'
            chkpts = ['unaligned', 'aligned']
        
        command = ''
        for chkpt in chkpts:
            command += f"""
echo "==========> Model {chkpt}:"
python generate_hypotheses.py -dn "{args.eval_dataset_name}" -mn "{args.model_name}" -hp "{hyperparameters}" -c "{chkpt}" -st "evaluation"
python evaluate.py -dn "{args.eval_dataset_name}" -mn "{args.model_name}" -hp "{hyperparameters}" -c "{chkpt}"
    """

    slurm_script = f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --hint=nomultithread
#SBATCH --time={args.max_time}
#SBATCH --output=slurm/out/{job_name}_%j.out

cd /gpfsdswork/projects/rech/hrc/urb81vh/translation-dpo
module purge
module load anaconda-py3/2023.09
conda activate tdpo

{command}
    """

    with open(f'slurm/{job_name}.slurm', 'w') as slurm_file:
        slurm_file.write(slurm_script)

    command = ['sbatch', f'slurm/{job_name}.slurm']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print("Standard Output:")
    print(stdout.decode())
    print("Standard Error:")
    print(stderr.decode())
    # os.remove(f'slurm/{job_name}.slurm')


if __name__ == '__main__':
    main()