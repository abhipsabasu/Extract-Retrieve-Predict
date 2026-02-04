import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--column', type=str,
                        help='Column name')
parser.add_argument('--dataset', type=str,
                        help='Dataset name')
parser.add_argument('--language', type=str, default=None,
                        help='Gre, Spa, Hin, Jap')
args = parser.parse_args()
column = args.column

nouns = ['apartment', 'bedroom', 'beach', 'flag', 'car', 'house', 'hotel', 'road', 'toilet', 'kitchen',
        'mountain', 'hill', 'lake', 'valley', 'island', 'shop', 'jewelry', 'office',
        'wedding dress', 'living room']
        
dfs = []
for noun in nouns:
    if args.language is None:
        df = pd.read_csv(f'data/distribution_{noun}_{args.column}_{args.dataset}.csv')
    else:
        df = pd.read_csv(f'data/distribution_{noun}_{args.column}_{args.dataset}_{args.language}.csv')
    df = df[df[args.column]!='no']
    dfs.append(df)
df_combined = pd.concat(dfs)
df_combined = df_combined.groupby(args.column)["counts"].sum().reset_index(name='counts')
df_combined = df_combined.sort_values(by='counts', ascending=False)
print(df_combined.head())
if args.language is None:
    df_combined.to_csv(f'data/distribution_combined_{args.column}_gemini_{args.dataset}.csv', index=False)
else:
    df_combined.to_csv(f'data/distribution_combined_{args.column}_gemini_{args.dataset}_{args.language}.csv', index=False)
