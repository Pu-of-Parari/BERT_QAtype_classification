import sys
import numpy as np
import pandas
import csv
import argparse
import os

import MeCab

"""交差検定用データ分割プログラム
args <<
--input: 全データのtsvファイル
--outputdir: 出力ディレクトリ
--k: 分割数

output >>
./outputdir/set_k/train,dev.tsv

"""

def wakati(input_str):
        '''分かち書き用関数
        input  << input_str : 入力テキスト
        output >> m.parse(wakatext) : 分かち済みテキスト'''
        wakatext = input_str
        #m = MeCab.Tagger('-Owakati')
        m = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/ipadic')#normal ipadic辞書指定
        #m = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')#neologd辞書指定
        #print(m.parse(wakatext))
        return m.parse(wakatext).replace("\n","")

def waka_file(csvfn):
    neo_fn = csvfn.replace(".tsv", "_neo.tsv")
    with open(csvfn,'r') as inf, open(neo_fn, 'w') as neof:
        org = csv.reader(inf, delimiter="\t", lineterminator="\n", )
        header = next(org)
        writer = csv.writer(neof, delimiter="\t", lineterminator="\n")
        writer.writerow(header)

        for line in org:
            sample = []
            #print(line)
            index = line[0]
            sentence_ = line[1]
            sentence = wakati(sentence_)
            label = line[2]
            #print(sentence)

            sample.append(index)
            sample.append(sentence)
            sample.append(label)
            writer.writerow(sample)

    return neo_fn



parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--outputdir', type=str)
parser.add_argument('--k', type=str, default=10)
parser.add_argument('--wakati', action='store_true') ##実行時に--wakatiを加えるとわかち処理

args = parser.parse_args()
target = args.input       # 分割対象のファイル名を取得
output_dir = args.outputdir
k_fold = int(args.k) # 分割の割合


if args.wakati:
    neo_target = waka_file(target)
    target = neo_target
    print("use >> ",target)



for i in range(0,k_fold):
    path = output_dir + "set_" + str(i)
    #print(path)
    os.makedirs(path, exist_ok=True)


# データの読み込み
df = pandas.read_table(target)
df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True) # ランダムに並べ替える

# 分割
p = int(len(df) / k_fold)


pt_list = []
for i in range(0,k_fold):
    idx_s = i*p
    idx_e = i*p +p
    print("set",i,"test use:",idx_s,"-", idx_e)
    tr_set_1 = df.iloc[:idx_s]
    tr_set_2 = df.iloc[idx_e:]
    tr_set = pandas.concat([tr_set_1,tr_set_2])
    output_dir_ = output_dir + "set_" + str(i) + "/"
    trnm = output_dir_ + "train.tsv"
    ts_set = df.iloc[idx_s:idx_e]

    tsnm = output_dir_ +"dev.tsv"
    tr_set.to_csv(trnm, sep='\t', index=False)
    ts_set.to_csv(tsnm, sep='\t', index=False)
    #print(len(tr_set), len(ts_set))
