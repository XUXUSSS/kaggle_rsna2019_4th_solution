model=$4
gpu=0
modelfold=$1
datafold=$2
ep=$3
tta=1
clip=1e-6
conf=./conf/${model}.py

snapshot=./model/${model}/fold${modelfold}_${ep}.pt
valid=./model/${model}/fold${modelfold}_${ep}_datafold${datafold}_tta${tta}.pkl
output=./features/${model}
output3d=./features3d/${model}

mkdir ./features
mkdir ./features3d
mkdir ${output}
mkdir ${output3d}

python -m src.cnn.main valid ${conf} --snapshot ${snapshot} --output ${valid} --n-tta ${tta} --fold ${datafold} --gpu ${gpu} --genfeat 1
python -m src.postprocess.analyse_features3d ${conf} --pkl ${valid} --output ${output} --fold ${modelfold} --datafold ${datafold} --output3d ${output3d}
