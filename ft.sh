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

data_dir=/scratch/jjery2243542/lip_reading/vctk
data_tar_path=/share/data/speech/jjery2243542/data/vctk/resampled_16khz/vctk_data.tar
feat_tar_path=/share/data/speech/jjery2243542/avhubert/features/vctk_feature.tar

if [[ ! -d $data_dir/data || ! -f $data_dir/finished ]]; then
    echo "untaring"
    if [ ! -d $data_dir ]; then
        mkdir -p $data_dir
    fi 
    tar -xf $data_tar_path -C $data_dir
    touch "$data_dir/finished" 
    echo "finishing untaring"
fi

if [[ ! -d $data_dir/features || ! -f $data_dir/feat_finished ]]; then
    echo "untaring features"
    if [ ! -d $data_dir ]; then
        mkdir -p $data_dir
    fi 
    tar -xf $feat_tar_path -C $data_dir
    mv $data_dir/vctk $data_dir/features
    touch "$data_dir/feat_finished" 
    echo "finishing untaring features"
fi

#modal=a_v_av
#size=base
model_dir=/share/data/speech/jjery2243542/lip2speech/av2wav/bs_32/lrs3_vc2/filtered_v2/1M/finetuned/${model}
list_dir=/share/data/speech/jjery2243542/avhubert/file_list/vctk
#root_dir=/share/data/lang/users/bshi/lip_reading/data
max_steps=50000
echo $model
trap 'echo signal recieved in BATCH!; kill -10 "${PID}"; wait "${PID}";' SIGUSR1

wav_dir=$list_dir/audio
feat_dir=$list_dir/features

valid_audio_list_dir=/share/data/speech/jjery2243542/avhubert/file_list/splits/audio/filtered_by_sisdr_23
valid_feat_list_dir=/share/data/speech/jjery2243542/avhubert/file_list/splits/features/filtered_by_sisdr_23

ckpt=/share/data/speech/jjery2243542/lip2speech/av2wav/bs_32/lrs3_vc2/filtered_v2/1M/large_lr_1e-4_warmup_xlarge2/weights-999999.pt
python -m wavegrad $model_dir --train_wav_file $wav_dir/train.txt --train_npy_files $feat_dir/train.txt $feat_dir/train.txt $feat_dir/train.txt --valid_wav_file $valid_audio_list_dir/valid.txt --valid_npy_files $valid_feat_list_dir/a/valid.txt $valid_feat_list_dir/v/valid.txt $valid_feat_list_dir/av/valid.txt --train_root_dir $data_dir --valid_root_dir $data_dir/../ --max_steps $max_steps --conf conf/finetune_vctk/${model}.yaml --fp16 --ckpt $ckpt &

PID="$!"
echo $PID
wait "${PID}"

