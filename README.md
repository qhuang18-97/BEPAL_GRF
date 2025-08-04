# Code of BEPAL for AAAI reproducibility Check
## Installation

First, clone the repo as BEPAL and install GRF which contains implementation for Predator-Prey and Traffic-Junction

```bash
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip
```
Install the GRF with multi-agent version (from https://github.com/CORE-Robotics-Lab/MAGIC)
```bash
git clone https://github.com/chrisyrniu/football.git
cd football
pip install .
```

Install the multi-agent environment wrapper for GRF:
```bash
cd envs/grf-envs
python setup.py develop
```

Next, install dependencies:
```bash
pip install -r requirements.txt
```

## Training
Training args for GRF:
```bash
python main.py --env_name grf --nagents 3 --nprocesses 1 --num_epochs 1000 --epoch_size 10 --hid_size 128 --detach_gap 10 --lrate 0.001 --value_coeff 0.01 --max_steps 80 --directed --gat_num_heads 1 --gat_hid_size 128 --gat_num_heads_out 1 --ge_num_heads 8 --use_gat_encoder --gat_encoder_out_size 32 --self_loop_type1 2 --self_loop_type2 2 --first_gat_normalize --second_gat_normalize --message_encoder --message_decoder --scenario academy_3_vs_1_with_keeper --num_controlled_lagents 3 --num_controlled_ragents 0 --reward_type scoring --recurrent
```
