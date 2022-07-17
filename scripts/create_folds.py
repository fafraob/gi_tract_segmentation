import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import pickle


def main():
    df = pd.read_csv('data/data.csv')
    fold_dir = 'data/folds'
    if not os.path.isdir(fold_dir):
        os.mkdir(fold_dir)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['unique_seg_flag'], groups=df["case_str"])):
        path = os.path.join(fold_dir, f'fold_{fold}.pickle')
        with open(path, 'wb') as f:
            obj = {
                'train_idx': train_idx.tolist(),
                'val_idx': val_idx.tolist()
            }
            pickle.dump(obj, f)


if __name__ == '__main__':
    main()
