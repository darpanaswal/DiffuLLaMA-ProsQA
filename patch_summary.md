# from repo root, diffu venv active
source diffu/bin/activate
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..

# generate datasets
python scripts/wic_to_diffusft.py --out_dir LLaMA-Factory/data --holdout_n 500 --seed 42

# build data + train with weight 10
ANSWER_LOSS_WEIGHT=10 bash train_wic.sh full