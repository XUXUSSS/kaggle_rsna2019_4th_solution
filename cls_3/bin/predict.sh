ep=best
gpu=0
tta=1
clip=1e-6

model=model001
for fold in 0 1 2 3 4
do 
	conf=./conf/${model}_${fold}.py
	snapshot=./model/${model}/fold${fold}_${ep}.pt

	for tta_id in 0 1 2 3 4
	do 
		output=./model/${model}/fold${fold}_${ep}_test_tta${tta}_${tta_id}.pkl
		submission=./data/submission/${model}_fold${fold}_${ep}_test_tta${tta}_${tta_id}.csv

		python -m src.cnn.main test ${conf} --snapshot ${snapshot} --output ${output} --n-tta ${tta} --fold ${fold} --gpu ${gpu} --ttaid ${tta_id} 
		python -m src.postprocess.make_submission --input ${test} --output ${submission} --clip ${clip} --sample_submission ../IFE_1/input/stage_2_sample_submission.csv
	done

done
