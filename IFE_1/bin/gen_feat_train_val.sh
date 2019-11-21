#! /bin/bash
ep=best
for modelfold in 0 1 2 3 4
do
	for model in model001 model002
	do
        for datafold in 0 1 2 3 4
        do
	        echo ${datafold}
            ./bin/gen_feat_train_val001.sh ${modelfold} ${datafold} ${ep} ${model}
        done
    done
done
