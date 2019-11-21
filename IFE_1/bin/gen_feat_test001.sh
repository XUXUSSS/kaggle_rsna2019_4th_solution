model=$1
gpu=0
fold=$2
ep=$3
tta=$4
clip=1e-6
conf=./conf/${model}.py

snapshot=./model/${model}/fold${fold}_${ep}.pt
test=./model/${model}/fold${fold}_${ep}_test_tta${tta}.pkl

python -m src.cnn.main test ${conf} --snapshot ${snapshot} --output ${test} --n-tta ${tta} --fold ${fold} --gpu ${gpu} --genfeat 1



output=./features/${model}
output3d=./features3d/${model}
mkdir ./features
mkdir ./features3d
mkdir ${output}
mkdir ${output3d}

for tta_id in 0 1 2 3 4
do
    python -m src.postprocess.analyse_features3d ${conf} --pkl ${test} --output ${output} --ttaid ${tta_id} --istest 1 --fold ${fold} --output3d ${output3d}
done

