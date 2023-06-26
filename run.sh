#!/usr/bin/zsh
#SBATCH --partition=speech-gpu
##SBATCH --partition=cpu
#SBATCH -c2
##SBATCH -C 2080ti
#SBATCH -o slurm/av2wav/slurm-%j.out
#SBATCH -e slurm/av2wav/slurm-%j.err
###SBATCH --open-mode=append
###SBATCH --exclude=gpu18
#SBATCH --signal=B:10@300

source ~/.zshrc
eval "$(conda shell.bash hook)"
conda activate avhubert

data_dir=/scratch/jjery2243542/lip_reading
data_tar_path=/share/data/lang/users/jjery2243542/lip_reading/data2.tar.gz
feat_tar_path=/share/data/speech/jjery2243542/avhubert/features/lrs3_vc2.tar
if [[ ! -d $data_dir/data || ! -f $data_dir/finished ]]; then
    echo "untaring"
    if [ ! -d $data_dir ]; then
        mkdir -p $data_dir
    fi 
    tar -xf $data_tar_path -C $data_dir
    touch "$data_dir/finished" 
    echo "finishing untaring"
fi

if [[ ! -d $data_dir/features || ! -f $data_dir/feat_v2_finished ]]; then
    # clean up the v1 features
    if [[ -d $data_dir/features ]]; then
        echo "rm the v1 features"
        rm -rf $data_dir/features
    fi
    echo "untaring features"
    if [ ! -d $data_dir ]; then
        mkdir -p $data_dir
    fi 
    tar -xf $feat_tar_path -C $data_dir
    mv $data_dir/lrs3_vc2 $data_dir/features
    touch "$data_dir/feat_v2_finished" 
    echo "finishing untaring features"
fi

#model=large_lr_1e-4_warmup
#size=base
model_dir=/share/data/speech/jjery2243542/lip2speech/av2wav/bs_32/lrs3_vc2/filtered_v2/1M/${model}
list_dir=/share/data/speech/jjery2243542/avhubert/file_list/splits
#root_dir=/share/data/lang/users/bshi/lip_reading/data
max_steps=1000000
echo $model
trap 'echo signal recieved in BATCH!; kill -10 "${PID}"; wait "${PID}";' SIGUSR1

wav_dir=$list_dir/audio/filtered_by_sisdr_23
feat_dir=$list_dir/features/filtered_by_sisdr_23
python -m wavegrad $model_dir --train_wav_file $wav_dir/train.txt --train_npy_files $feat_dir/a/train.txt $feat_dir/v/train.txt $feat_dir/av/train.txt --valid_wav_file $wav_dir/valid.txt --valid_npy_files $feat_dir/a/valid.txt $feat_dir/v/valid.txt $feat_dir/av/valid.txt --train_root_dir $data_dir --valid_root_dir $data_dir --max_steps $max_steps --conf conf/${model}.yaml --fp16 &

PID="$!"
echo $PID
wait "${PID}"

