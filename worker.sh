learning_rate=$1
latent_dim=$2
beta=$3

PARENT_DIR="$(dirname $PWD)"
EXEC_DIR=$PWD
log_dir="logs/lr=${learning_rate}_latent=${latent_dim}_beta=${beta}"
echo "Current working directory is: $(pwd)"
python main.py --n 10 --lr ${learning_rate} --latent ${latent_dim} --beta ${beta} --log_dir=${log_dir} --vae --checkpoint checkpoints/vae-n25-beta0.pt