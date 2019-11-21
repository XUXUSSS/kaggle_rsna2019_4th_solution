model=model001
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

mkdir ./features
mkdir ${output}

python -m src.cnn.main valid ${conf} --snapshot ${snapshot} --output ${valid} --n-tta ${tta} --fold ${datafold} --gpu ${gpu} --genfeat 1
python -m src.postprocess.analyse_features ${conf} --pkl ${valid} --output ${output} --fold ${modelfold} --datafold ${datafold}
