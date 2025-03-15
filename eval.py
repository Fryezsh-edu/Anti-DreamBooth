import logging
import subprocess
import os
import torch
from tqdm.auto import tqdm
from evaluations.ism_fdfr import matching_score_genimage_id
from email_sender import send_email

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '2'

path = "/home/yuzh/Anti-DreamBooth"
data_path = os.path.join(path, "data")
output_path = os.path.join(path, "outputs/ASPL")

attack_script = '''
python attacks/aspl.py \
    --pretrained_model_name_or_path=./stable-diffusion/stable-diffusion-2-1-base \
    --enable_xformers_memory_efficient_attention \
    --instance_data_dir_for_train=data/{id}/set_A \
    --instance_data_dir_for_adversarial=data/{id}/set_B \
    --instance_prompt="a photo of sks person" \
    --class_data_dir=data/class-person \
    --num_class_images=200 \
    --class_prompt="a photo of person" \
    --output_dir=outputs/ASPL/{id}_ADVERSARIAL \
    --center_crop \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --resolution=512 \
    --train_text_encoder \
    --train_batch_size=1 \
    --max_f_train_steps=3 \
    --max_adv_train_steps=6 \
    --max_train_steps=50 \
    --checkpointing_iterations=10 \
    --learning_rate=5e-7 \
    --pgd_alpha=5e-3 \
    --pgd_eps=5e-2
'''

dreambooth_script = '''
python train_dreambooth.py \
    --pretrained_model_name_or_path=./stable-diffusion/stable-diffusion-2-1-base \
    --enable_xformers_memory_efficient_attention \
    --train_text_encoder \
    --instance_data_dir=outputs/ASPL/{id}_ADVERSARIAL/noise-ckpt/10 \
    --class_data_dir=data/class-person \
    --output_dir=outputs/ASPL/{id}_DREAMBOOTH \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --instance_prompt="a photo of sks person" \
    --class_prompt="a photo of person" \
    --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-7 \
    --lr_scheduler=constant \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --center_crop \
    --mixed_precision=bf16 \
    --prior_generation_precision=bf16 \
    --sample_batch_size=8
'''

need_metrics = True
need_send_email = True

prompts = ["a_dslr_portrait_of_sks_person", "a_photo_of_sks_person"]
result = {
    "a_dslr_portrait_of_sks_person": {
            "ism": [],
            "fdfr": []
        },
    "a_photo_of_sks_person": {
            "ism": [],
            "fdfr": []
    }
}

version = "bv0.6"
log_name = version + ".log"
logging.basicConfig(
    filename=log_name,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

files = os.listdir(data_path)
files_sorted = sorted(files)

cnt = 0


try:
    for idx in tqdm(files_sorted):
        if "n0" not in idx:
            continue
        if not os.path.exists(os.path.join(output_path, f"{idx}_ADVERSARIAL")):
            subprocess.run(attack_script.format(id=idx), shell=True, env=env)
            subprocess.run(dreambooth_script.format(id=idx), shell=True, env=env)
            cnt += 1
            if need_metrics:
                for prompt in prompts:
                    idx_data_dir = os.path.join(data_path, idx, "set_A"), os.path.join(data_path, idx, "set_B")
                    mid_dir = "{}_DREAMBOOTH/checkpoint-1000/dreambooth".format(idx)
                    idx_fake_dir = os.path.join(output_path, mid_dir, prompt)
                    ism, fdfr = matching_score_genimage_id(idx_fake_dir, idx_data_dir)
                    result[prompt]["fdfr"].append(fdfr)
                    if ism is None:
                        continue
                    result[prompt]["ism"].append(ism)
                    mean_fdfr = torch.mean(torch.tensor(result[prompt]["fdfr"]))
                    mean_ism = torch.mean(torch.stack(result[prompt]["ism"]))
                    logging.info(f'{idx} "{prompt}": fdfr({fdfr:.4f}/{mean_fdfr:.4f}) ism({ism:.4f}/{mean_ism:.4f})')
        
                if need_send_email and (cnt % 10 == 0):
                    content = ""
                    for prompt in prompts:
                        mean_fdfr = torch.mean(torch.tensor(result[prompt]["fdfr"]))
                        mean_ism = torch.mean(torch.stack(result[prompt]["ism"]))
                        content += f'"{prompt}" mean_fdfr:{mean_fdfr}, mean_ism:{mean_ism}\n'
                    send_email("zhyu_edu@163.com", f"[Anti-DreamBooth {version} {cnt}/50] Results", content, log_name)
                    
except Exception as e:
    send_email("zhyu_edu@163.com", f"[Anti-DreamBooth {version} {cnt}/50] Error", "实验异常中断", log_name)
    print(e)
