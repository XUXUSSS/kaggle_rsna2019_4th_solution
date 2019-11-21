gpu=0
ep=best
tta=5
model=model001

for fold in 0 1 2 3 4
do
	sh ./bin/gen_feat_test001.sh ${model} ${fold} ${ep} ${tta}
done



