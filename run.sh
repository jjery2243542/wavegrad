#!/usr/bin/bash
#SBATCH --partition=speech-gpu
#SBATCH -c8
##SBATCH -C 2080ti
#SBATCH -o slurm/av2wav/slurm-%j.out
#SBATCH -e slurm/av2wav/slurm-%j.err
###SBATCH --exclude=gpu18
#SBATCH --signal=B:10@300

source ~/.zshrc
eval "$(conda shell.bash hook)"
conda activate avhubert

data_dir=/scratch/jjery2243542/lip_reading
tar_path=/share/data/lang/users/jjery2243542/lip_reading/data2.tar.gz
if [[ ! -d $data_dir || ! -f $data_dir/finished ]]; then
    echo "untaring"
    if [ ! -d $data_dir ]; then
        mkdir -p $data_dir
    fi 
    tar -xf $tar_path -C $data_dir
    touch "$data_dir/finished" 
    echo "finishing untaring"
fi

#modal=a_v_av
#size=base
data_dir=$data_dir/data
model_dir=/share/data/speech/jjery2243542/lip2speech/av2wav/bs_32/lrs3_vc2/diff_cond/${modal}_${size}
#model_dir=/share/data/speech/jjery2243542/lip2speech/av2wav/bs_32/lrs3_vc2/diff_cond/test
list_dir=/share/data/speech/jjery2243542/avhubert/file_list/splits
#root_dir=/share/data/lang/users/bshi/lip_reading/data
max_steps=500000

trap 'echo signal recieved in BATCH!; kill -10 "${PID}"; wait "${PID}";' SIGUSR1

python -m wavegrad $model_dir --train_wav_files $list_dir/audio/train.txt --train_mp4_files $list_dir/video/train.txt --valid_wav_files $list_dir/audio/valid.txt --valid_mp4_files $list_dir/video/valid.txt --root_dir $data_dir --max_steps $max_steps --conf conf/${size}_${modal}.yaml --fp16 & 

PID="$!"
echo $PID
wait "${PID}"

