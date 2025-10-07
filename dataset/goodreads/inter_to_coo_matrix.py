import pandas as pd
import numpy as np
import pickle
from scipy.sparse import coo_matrix

INPUT_TSV = 'goodreads.inter'
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1
USER_COL = 'userId:token'
ITEM_COL = 'movieId:token'
RATING_COL = 'rating:float'
TIME_COL = 'timestamp:float'

df = pd.read_csv(INPUT_TSV, sep='\t')

user_tokens  = df[USER_COL].unique()
movie_tokens = df[ITEM_COL].unique()
user_to_idx  = {u: i for i, u in enumerate(user_tokens)}
movie_to_idx = {m: i for i, m in enumerate(movie_tokens)}
n_users, n_movies = len(user_tokens), len(movie_tokens)

def split_user(df_user):
    df_sorted = df_user.sort_values(TIME_COL)
    n = len(df_sorted)
    n_train = int(np.floor(TRAIN_FRAC * n))
    n_val   = int(np.floor(VAL_FRAC   * n))
    n_test  = n - n_train - n_val
    return (
        df_sorted.iloc[:n_train],
        df_sorted.iloc[n_train:n_train + n_val],
        df_sorted.iloc[n_train + n_val:]
    )

train_list, val_list, test_list = [], [], []
for _, grp in df.groupby(USER_COL):
    tr, va, te = split_user(grp)
    train_list.append(tr); val_list.append(va); test_list.append(te)

df_train = pd.concat(train_list)
df_val   = pd.concat(val_list)
df_test  = pd.concat(test_list)

def build_coo(sub_df):
    rows = sub_df[USER_COL].map(user_to_idx).to_numpy()
    cols = sub_df[ITEM_COL].map(movie_to_idx).to_numpy()
    data = sub_df[RATING_COL].to_numpy()
    return coo_matrix((data, (rows, cols)),
                      shape=(n_users, n_movies),
                      dtype=float)

for split_name, subset in [('trn', df_train), ('val', df_val), ('tst', df_test)]:
    mat = build_coo(subset)
    fname = f'{split_name}Mat.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(mat, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Wrote {fname}: shape={mat.shape}, nnz={mat.nnz}')
