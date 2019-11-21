mkdir -p cache model data/submission

# train
python -m src.preprocess.dicom_to_dataframe --input ./input/stage_2_train.csv --output ./cache/train_raw.pkl --imgdir ./input/stage_2_train_images
python -m src.preprocess.create_dataset --input ./cache/train_raw.pkl --output ./cache/train.pkl

for seed in 10 25 50 75 100
do
	python -m src.preprocess.make_folds --input ./cache/train.pkl --output ./cache/train_folds_s${seed}.pkl --n-fold 5 --seed ${seed}
done

ln -s ../IFE_1/cache ../IFE_2/
ln -s ../IFE_1/cache ../IFE_3/
ln -s ../IFE_1/cache ../cls_1/
ln -s ../IFE_1/cache ../cls_2/
ln -s ../IFE_1/cache ../cls_3/

# test
python -m src.preprocess.dicom_to_dataframe --input ./input/stage_2_sample_submission.csv --output ./cache/test_raw.pkl --imgdir ./input/stage_2_test_images
python -m src.preprocess.create_dataset --input ./cache/test_raw.pkl --output ./cache/test.pkl
