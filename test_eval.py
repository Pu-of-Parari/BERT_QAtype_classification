import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

"""run_classifier_Atype.pyのtest出力結果の評価
input << "./test_results.tsv" :モデル出力ファイル
         "./atype_test.tsv" :参照用テストデータセット
output >> accuracy(標準出力)
          confusion matrix(標準出力)
"""

def predLoader(pred_file):
    with open(pred_file) as pf:
        pred_data_list = []
        for i, line_ in enumerate(pf):
            if i == 0:
                labels = line_.replace("\n","").split("\t")
                continue

            line = line_.replace("\n","").split("\t")

            idx = line[0]
            pred_p = []
            pred_data = []

            for p in line[1:]: pred_p.append(float(p))
            #print(pred_p)
            label_idx = pred_p.index(max(pred_p))
            label = labels[label_idx+1]
            #print(idx, label)
            pred_data.append(idx)
            pred_data.append(label)

            pred_data_list.append(pred_data)

    return pred_data_list


def testDatasetLoader(test_file):
    with open(test_file) as tstf:
        inf_data_list = []
        for i, line_ in enumerate(tstf):
            if i == 0:
                #labels = line_.replace("\n","").split("\t")
                continue
            line = line_.replace("\n","").split("\t")
            idx = line[0]
            sentence = line[1]
            label = line[2]
            inf_data = []

            inf_data.append(idx)
            inf_data.append(sentence)
            inf_data.append(label)

            inf_data_list.append(inf_data)

    return inf_data_list


def evaluation(predictions_data, inferences_data):
    count = 0
    pos, neg = 0, 0
    for pred, inf in zip(predictions_data, inferences_data):
        count += 1
        p_label = pred[1]
        i_label = inf[2]

        if p_label == i_label:
            pos += 1
        else:
            neg += 1

    acc = pos / count
    total = pos + neg
    print('acc: %f(%d/%d)' % (acc, pos, total))



def print_cmx(y_pred_, y_true_):
    #labels = sorted(list(set(y_true_)))

    labels_a = ["sent","phrase","word","num","other"]
    #labels_q = ["慣習・経験・推奨","要点・注意点","方法","特徴・定義・事実","理由・原因"]
    #labels_q = ["exp","point","how","fact","reason"]
    cmx_data = confusion_matrix(y_true_, y_pred_, labels=labels_a)

    df_cmx = pd.DataFrame(cmx_data, index=labels_a, columns=labels_a)

    plt.figure(figsize = (10,7))
    plt.title("References A_type vs Predictions A_type")
    sns.heatmap(df_cmx, annot=True,cmap=plt.cm.Blues)
    plt.xlabel("Predictions A_type")
    plt.ylabel("References A_type")
    plt.show()




if __name__ == "__main__":
    p_data = predLoader("./test_results.tsv")
    i_data = testDatasetLoader("./atype_test.tsv")

    evaluation(p_data, i_data)

    p_label = []
    for p in p_data: p_label.append(p[1])
    i_label = []
    for i in i_data: i_label.append(i[2])

    print_cmx(p_label, i_label)
