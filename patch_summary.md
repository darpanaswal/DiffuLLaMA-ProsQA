# from repo root, diffu venv active
source diffu/bin/activate
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..

bash fix_launcher_patch.sh          # already in repo — 4d attn mask
bash fix_torchrun_patch.sh          # torchrun -> python -m torch.distributed.run
bash fix_answer_upweight_patch.sh   # answer-token loss upweight

# edit dataset_info.json (add wic_train/wic_valid entries above)

# build data + train with weight 10
ANSWER_LOSS_WEIGHT=10 bash train_wic.sh full