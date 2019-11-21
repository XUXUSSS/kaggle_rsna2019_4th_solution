gpu=0
ep=best
tta=5


for model in model001 model002
do
	for fold in 0 1 2 3 4
	do
        sh ./bin/gen_feat_test001.sh ${model} ${fold} ${ep} ${tta}
    done
done


