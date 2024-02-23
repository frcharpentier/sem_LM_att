import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sbn

def plot_conf_mtx(confusion, label_names):
    #mini = np.min(confusion, axis=1).reshape(-1,1)
    #print(mini.shape)
    #maxi = np.max(confusion, axis=1).reshape(-1,1)
    confusion_norm = (confusion)/(np.sum(confusion, axis=1).reshape(-1,1))
    fig, axes = plt.subplots(figsize=(16, 14))
    sbn.heatmap(
        confusion_norm,
        annot=confusion,
        cbar=False,
        fmt="d",
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="viridis",
        ax=axes
    )
    axes.set_xlabel("Prédictions")
    axes.set_ylabel("Réalité")
    plt.show()



def confusion_to_truth_pred(conf):
    sh = conf.shape
    assert sh[0] == sh[1]
    nb_classes = sh[0]
    len_ds = np.sum(conf)
    realite = np.zeros((len_ds,), dtype=np.uint8)
    predictions = np.zeros((len_ds,), dtype=np.uint8)
    i = 0
    for R in range(nb_classes):
        for P in range(nb_classes):
            n = conf[R,P]
            realite[i:i+n] = R
            predictions[i:i+n] = P
            i += n

    return realite, predictions

def conf_to_n_conf2(conf):
    sh = conf.shape
    assert sh[0] == sh[1]
    nb_classes = sh[0]
    mask = 1-np.eye(nb_classes)
    faux = conf * mask
    TP = conf.diagonal()
    FN = faux.sum(axis=1)
    FP = faux.sum(axis=0)
    samples = conf.sum()    
    resu = []
    for C in range(nb_classes):
        TN = samples - TP[C] - FN[C] - FP[C]
        conf2 = {"TP" : TP[C], "FN" : FN[C], "FP": FP[C], "TN" : TN}
        resu.append(conf2)
    return resu

def conf_to_n_stats(conf, classes):
    sh = conf.shape
    assert sh[0] == sh[1]
    assert sh[0] == len(classes)
    n_conf2 = conf_to_n_conf2(conf)
    P = {cl : prec(cnf) for cl, cnf in zip(classes, n_conf2)}
    R = {cl : rappel(cnf) for cl, cnf in zip(classes, n_conf2)}
    F1 = { cl: F1_score(cnf) for cl, cnf in zip(classes, n_conf2)}
    return P, R, F1

def conf_to_acc_bal_accuracy(conf):
    sh = conf.shape
    assert sh[0] == sh[1]
    nb_classes = sh[0]
    mask = np.eye(nb_classes)
    corrects = (conf * mask).sum()
    samples = conf.sum()
    acc = corrects / samples
    bal_acc = np.array([rappel(C2) for C2 in conf_to_n_conf2(conf)]).mean()
    return acc, bal_acc

def prec(conf2):
    return conf2["TP"] / (conf2["TP"] + conf2["FP"])

def rappel(conf2):
    return conf2["TP"] / (conf2["TP"] + conf2["FN"])

def exactitud(conf2):
    return (conf2["TP"]+conf2["TN"])/(conf2["TP"]+conf2["FN"]+conf2["FP"]+conf2["TN"])

def F1_score(conf2):
    return 2*conf2["TP"]/(2*conf2["TP"] + conf2["FP"] + conf2["FN"])

def calcul_Prec_Rappel(conf1, conf2):
    classes = ["ARG1", "<no-rel>", "ARG0", "topic", "poss", "mod",
               "location", "quant", "ARG2", "time", "purpose", "part",
               "ARG3", "<other>", "ARG4", "condition", "manner", "degree"]
    acc1, bal_acc1 = conf_to_acc_bal_accuracy(conf1)
    print("Accuracy 1: %f, balanced accuracy 1: %f"%(acc1, bal_acc1))
    acc2, bal_acc2 = conf_to_acc_bal_accuracy(conf2)
    print("Accuracy 2: %f, balanced accuracy 2: %f"%(acc2, bal_acc2))
    print()
    P1, R1, F1_1 = conf_to_n_stats(conf1, classes)
    P2, R2, F1_2 = conf_to_n_stats(conf2, classes)
    dico = {K: (P1[K], R1[K], P2[K], R2[K]) for K in classes}
    for k, t in dico.items():
        print("%s & %.02f & %.02f & %.02f & %.02f \\\\"%(k, t[0], t[1], t[2], t[3]))

def calcul_F1_RoB_GPT2(confR, confG):
    classes = ["ARG1", "<no-rel>", "ARG0", "topic", "poss", "mod",
               "location", "quant", "ARG2", "time", "purpose", "part",
               "ARG3", "<other>", "ARG4", "condition", "manner", "degree"]
    acc1, bal_acc1 = conf_to_acc_bal_accuracy(confR)
    print("Accuracy RoBERTa: %f, balanced accuracy RoBERTa: %f"%(acc1, bal_acc1))
    acc2, bal_acc2 = conf_to_acc_bal_accuracy(confG)
    print("Accuracy GPT2: %f, balanced accuracy GPT2: %f"%(acc2, bal_acc2))
    print()
    P1, R1, F1_1 = conf_to_n_stats(confR, classes)
    P2, R2, F1_2 = conf_to_n_stats(confG, classes)
    dico = {K: (P1[K], R1[K], F1_1[K], P2[K], R2[K], F1_2[K]) for K in classes}
    for k, t in dico.items():
        print("%s & %.02f & %.02f & %.02f & %.02f \\\\"%(k, t[0], t[1], t[3], t[4]))


def calcul_F1_RoB_GPT_GPTa(confR, confG, confGa, classes):
    acc1, bal_acc1 = conf_to_acc_bal_accuracy(confR)
    print("Accuracy RoBERTa: %f, balanced accuracy RoBERTa: %f"%(acc1, bal_acc1))
    acc2, bal_acc2 = conf_to_acc_bal_accuracy(confG)
    print("Accuracy GPT2: %f, balanced accuracy GPT2: %f"%(acc2, bal_acc2))
    acc3, bal_acc3 = conf_to_acc_bal_accuracy(confGa)
    print("Accuracy GPT2: %f, balanced accuracy GPT2: %f"%(acc3, bal_acc3))
    print()
    P1, R1, F1_1 = conf_to_n_stats(confR, classes)
    P2, R2, F1_2 = conf_to_n_stats(confG, classes)
    P3, R3, F1_3 = conf_to_n_stats(confGa, classes)
    dico = {K: (F1_1[K], F1_2[K], F1_3[K]) for K in classes}
    l_R = [(R, t[0]) for R, t in dico.items()]
    l_G = [(R, t[1]) for R, t in dico.items()]
    l_Ga = [(R, t[2]) for R, t in dico.items()]
    l_R.sort(key = lambda x: x[1])
    l_G.sort(key = lambda x: x[1])
    l_Ga.sort(key = lambda x: x[1])

    print(l_R)
    print(l_G)
    print(l_Ga)

    #for k, t in dico.items():
    for k, _ in l_R[::-1]:
        t = dico[k]
        print("%s & %.02f & %.02f & %.02f \\\\"%(k, t[0], t[1], t[2]))
    

    

if __name__ == "__main__":
    
    conf_QK_roberta = np.array([[2294, 45, 57, 31, 292, 139, 143, 88, 93, 281, 89,
255, 27, 85, 259, 79, 50, 21],
[ 225, 6175, 1283, 868, 653, 60, 473, 274, 185, 54, 158,
135, 302, 90, 35, 747, 217, 350],
[ 62, 440, 950, 138, 165, 17, 126, 141, 73, 28, 130,
62, 74, 17, 18, 249, 39, 208],
[ 51, 292, 212, 4374, 70, 32, 210, 23, 113, 12, 81,
61, 110, 22, 8, 109, 299, 37],
[ 6, 6, 6, 1, 165, 2, 13, 10, 4, 1, 7,
2, 7, 2, 0, 13, 4, 2],
[ 43, 10, 7, 3, 8, 775, 25, 15, 29, 5, 38,
30, 80, 6, 6, 12, 6, 7],
[ 176, 239, 232, 279, 177, 232, 3158, 191, 279, 119, 161,
130, 298, 189, 47, 161, 154, 110],
[ 16, 5, 2, 0, 24, 1, 6, 201, 3, 1, 4,
10, 3, 1, 0, 9, 0, 7],
[ 38, 22, 25, 22, 50, 66, 62, 20, 168, 11, 69,
56, 47, 27, 8, 47, 11, 56],
[ 19, 1, 1, 1, 0, 1, 10, 2, 10, 464, 2,
3, 0, 7, 23, 3, 1, 0],
[ 12, 4, 12, 6, 7, 14, 14, 4, 24, 4, 388,
9, 7, 14, 0, 15, 6, 24],
[ 23, 3, 11, 3, 3, 5, 6, 18, 16, 4, 11,
267, 9, 3, 21, 12, 1, 9],
[ 0, 5, 2, 6, 3, 8, 10, 6, 4, 1, 3,
6, 133, 1, 1, 3, 0, 0],
[ 1, 2, 0, 0, 0, 0, 3, 2, 1, 0, 0,
1, 3, 144, 0, 0, 0, 0],
[ 9, 0, 0, 2, 0, 4, 4, 0, 6, 8, 1,
23, 1, 3, 291, 1, 0, 0],
[ 9, 32, 27, 9, 21, 3, 15, 9, 13, 4, 14,
13, 7, 5, 4, 86, 4, 29],
[ 3, 2, 1, 13, 11, 2, 5, 0, 2, 0, 6,
0, 2, 5, 1, 4, 371, 0],
[ 1, 3, 3, 0, 4, 1, 1, 6, 4, 1, 8,
5, 0, 0, 0, 8, 0, 66]])
    

    
    
    conf_QK_GPT2 = np.array([[1621, 95, 46, 254, 282, 179, 131, 64, 47, 389, 102,
196, 33, 237, 311, 73, 204, 64],
[ 829, 2675, 653, 1115, 702, 625, 696, 452, 192, 268, 447,
689, 558, 372, 336, 545, 429, 701],
[ 89, 210, 470, 246, 162, 149, 183, 199, 52, 74, 164,
150, 112, 83, 60, 164, 101, 269],
[ 135, 108, 182, 3661, 154, 125, 309, 147, 60, 71, 151,
87, 166, 181, 34, 52, 454, 39],
[ 22, 9, 4, 3, 124, 3, 14, 14, 3, 1, 13,
10, 7, 7, 0, 6, 8, 3],
[ 45, 26, 21, 57, 18, 455, 68, 51, 14, 21, 71,
66, 94, 17, 20, 19, 12, 30],
[ 90, 216, 250, 525, 272, 273, 2239, 208, 151, 151, 265,
133, 489, 262, 79, 156, 428, 145],
[ 11, 13, 3, 6, 31, 12, 8, 147, 2, 2, 8,
6, 13, 7, 0, 10, 3, 11],
[ 39, 28, 28, 48, 60, 65, 56, 30, 57, 24, 83,
31, 65, 43, 12, 37, 28, 71],
[ 27, 5, 3, 18, 5, 9, 17, 0, 6, 400, 3,
4, 2, 14, 21, 2, 10, 2],
[ 21, 8, 10, 24, 26, 29, 23, 10, 15, 8, 241,
27, 10, 39, 2, 15, 24, 32],
[ 23, 19, 13, 12, 12, 25, 13, 28, 11, 1, 14,
164, 16, 6, 34, 16, 4, 14],
[ 1, 4, 2, 6, 4, 11, 10, 8, 1, 0, 3,
8, 131, 0, 0, 1, 1, 1],
[ 9, 0, 1, 7, 10, 0, 4, 3, 1, 3, 10,
2, 0, 87, 1, 1, 13, 5],
[ 14, 3, 1, 5, 5, 7, 4, 1, 1, 12, 4,
14, 1, 8, 270, 0, 1, 2],
[ 15, 12, 11, 12, 30, 10, 20, 17, 6, 13, 10,
18, 11, 9, 4, 55, 9, 42],
[ 11, 1, 2, 32, 27, 5, 13, 1, 7, 6, 15,
3, 2, 62, 1, 2, 237, 1],
[ 4, 3, 3, 3, 7, 6, 1, 3, 4, 1, 9,
4, 1, 2, 0, 2, 0, 58]])
    
    conf_QK_GPT2_CSSC = np.array([[1981, 42, 72, 94, 295, 159, 178, 82, 76, 344, 88,
231, 16, 175, 277, 64, 120, 34],
[ 176, 5140, 1272, 1085, 683, 137, 565, 407, 199, 54, 227,
262, 427, 133, 51, 702, 372, 392],
[ 58, 397, 850, 165, 183, 38, 150, 163, 89, 18, 130,
85, 91, 52, 16, 202, 55, 195],
[ 41, 283, 232, 4181, 99, 63, 175, 46, 126, 17, 104,
52, 126, 51, 14, 100, 365, 41],
[ 20, 9, 6, 2, 142, 1, 12, 14, 9, 1, 7,
2, 4, 1, 0, 8, 9, 4],
[ 46, 9, 12, 36, 8, 687, 54, 27, 17, 8, 45,
36, 72, 10, 5, 13, 10, 10],
[ 150, 289, 356, 332, 195, 226, 2449, 203, 211, 134, 181,
180, 372, 363, 62, 176, 348, 105],
[ 20, 5, 12, 3, 25, 1, 6, 171, 6, 0, 8,
7, 5, 3, 1, 10, 0, 10],
[ 32, 23, 34, 34, 41, 59, 59, 24, 133, 20, 84,
44, 49, 21, 10, 50, 35, 53],
[ 22, 1, 0, 3, 1, 2, 11, 1, 8, 454, 2,
4, 1, 11, 16, 4, 6, 1],
[ 15, 8, 11, 10, 11, 23, 22, 6, 30, 2, 333,
13, 3, 13, 2, 11, 15, 36],
[ 24, 4, 6, 14, 3, 11, 10, 19, 19, 2, 9,
228, 13, 6, 29, 17, 3, 8],
[ 1, 10, 1, 4, 4, 5, 17, 6, 7, 0, 4,
6, 121, 1, 0, 2, 0, 3],
[ 3, 1, 0, 0, 0, 0, 3, 0, 4, 1, 3,
0, 0, 136, 0, 0, 6, 0],
[ 12, 1, 3, 1, 2, 1, 6, 1, 3, 12, 1,
12, 0, 3, 292, 1, 2, 0],
[ 5, 23, 18, 8, 26, 8, 11, 11, 10, 11, 9,
14, 8, 3, 3, 96, 9, 31],
[ 9, 1, 0, 15, 21, 2, 10, 0, 4, 2, 6,
1, 0, 7, 1, 3, 344, 2],
[ 0, 6, 10, 3, 2, 3, 0, 5, 4, 0, 10,
6, 2, 0, 0, 5, 0, 55]])
    
    classes = [ ":mod",":ARG1",":ARG2",":ARG0",":topic",":time",
                "<no_rel>",":purpose","<other>",":quant",":location",
                ":manner",":condition",":part",":degree",":ARG3",":poss",
                ":ARG4"]
    #plot_conf_mtx(conf_QK_roberta, classes)
    
    print("Résultats :")
    calcul_F1_RoB_GPT_GPTa(conf_QK_roberta, conf_QK_GPT2, conf_QK_GPT2_CSSC, classes)
    #print("Résultats pour GPT2 :")
    #calcul(conf_att_GPT2, conf_QK_GPT2)
    print()
    print("##########")

