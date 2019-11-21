#! /bin/bash

outlist=./list/submit.list
outcsv=./csv/final_submit.csv
rm ${outlist}
find ../cls_1/data/submission -name '*.csv' >> ${outlist}
find ../cls_2/data/submission -name '*.csv' >> ${outlist}
find ../cls_3/data/submission -name '*.csv' >> ${outlist}
python P00_ensemble_csv.py --input ${outlist} --output ${outcsv}
