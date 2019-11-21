model=$1
gpu=0
fold=$2
ep=$3
tta=$4
clip=1e-6
conf=./conf/${model}.py

snapshot=./model/${model}/fold${fold}_${ep}.pt
test=./model/${model}/fold${fold}_${ep}_test_tta${tta}.pkl
sub=./data/submission/${model}_fold${fold}_${ep}_test_tta${tta}.csv

python -m src.cnn.main test ${conf} --snapshot ${snapshot} --output ${test} --n-tta ${tta} --fold ${fold} --gpu ${gpu} --genfeat 1



output=./features/${model}
mkdir ./features
mkdir ${output}

for tta_id in 0 1 2 3 4
do

python -m src.postprocess.analyse_features ${conf} --pkl ${test} --output ${output} --ttaid ${tta_id} --istest 1 --fold ${fold}

done

