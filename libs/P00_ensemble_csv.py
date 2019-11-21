import pandas as pd
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',help='Input submission list')
    parser.add_argument('--output',help='Output submission csv')
    parser.add_argument('--score',default=0.001,type=float)
    parser.add_argument('--gmean',default=False, type=float)
    return parser.parse_args()


def main():
    args = get_args()
    list_path = args.input
    with open(list_path,'r') as f:
        lines =  f.readlines()


    cells = []
    for line in lines:
        line = line.strip()
        csv_path = line
        df = pd.read_csv(csv_path)
        labels= df['Label'].to_numpy()
        cells.append(labels)


    if not args.gmean:
        df['Label'] = np.mean(cells,axis=0)
    else:
        df['Label'] = np.power(np.mean(np.power(np.array(cells),args.gmean), axis=0),1/args.gmean)
    #out_path = "{}_merge.csv".format(csv_path[:-4])
    out_path = args.output
    print(out_path)

    df.to_csv(out_path,index=None)


if __name__ == '__main__':
    main()
