git submodule init
cd m2d
curl -o util/lars.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lars.py
curl -o util/lr_decay.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_decay.py
curl -o util/lr_sched.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_sched.py
curl -o util/misc.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/misc.py
curl -o m2d/pos_embed.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py
curl -o train_audio.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o train_speech.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o mae_train_audio.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o m2d/engine_pretrain_m2d.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/engine_pretrain.py
curl -o m2d/models_mae.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py
curl -o m2d/timm_layers_pos_embed.py https://raw.githubusercontent.com/huggingface/pytorch-image-models/e9373b1b925b2546706d78d25294de596bad4bfe/timm/layers/pos_embed.py
patch -p1 < patch_m2d.diff
wget https://github.com/nttcslab/m2d/releases/download/v0.1.0/m2d_vit_base-80x608p16x16-221006-mr6.zip
unzip m2d_vit_base-80x608p16x16-221006-mr6.zip
pip install -r requirements.txt