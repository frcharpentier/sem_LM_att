import sys
import traceback
import random
import pandas as pnd
import numpy as np
import joblib
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sbn

#import json
import tqdm
from scipy.sparse import dok_array
#from sklearn import datasets
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble._forest import _generate_sample_indices as sample_function
from sklearn.ensemble import _forest
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from sklearn.model_selection import train_test_split

import plotly.offline
import plotly.graph_objs as gr_ob
import plotly.express as px

import logging
from report_generator import HTML_REPORT

class RandoFo_Summary():
    def __init__(self, RF, Feuilles=None):
        sampler = RF._custom_sample
        self.n_samples = sampler.n_samples
        self.y = sampler.y
        self.dic_list = sampler.dic_list
        self.nb_classes = sampler.nb_classes
        self.oob_score = RF.oob_score_
        if Feuilles == None:
           logging.info("Calcul matrice prox")
           Feuilles = compute_proximity(RF, sampler.X, output_leaves=True)
        self.F = Feuilles


class RandoFo_Custom_Bootstrap(RandomForestClassifier):
    
    def set_sampler(self, fonction):
        self._custom_sample = fonction

    def fit(self, X, y, sample_weight=None):
        if hasattr(self, "_custom_sample"):
            _forest.__dict__["_generate_sample_indices"] = self._custom_sample
        resu = super().fit(X, y, sample_weight)
        _forest.__dict__["_generate_sample_indices"] = sample_function
        return resu
    



class DS_sampler:
    def __init__(self, DataFrame_X, Series_y):
        ns = DataFrame_X.shape[0]
        assert ns == Series_y.shape[0]
        self.n_samples = ns
        self.y, self.dic_list = pnd.factorize(Series_y)
        #transforme les labels en des numéros uniques
        self.X = DataFrame_X.to_numpy()
        self.indices_ordre = np.argsort(self.y)
        #donne le numéro des indices nécessaire pour avoir y classé dans l’ordre
        self.nb_classes = len(self.dic_list)
        self.range_classes = np.arange(self.nb_classes)
        self.effectifs = np.bincount(self.y, minlength=self.nb_classes)
        cumsum = np.cumsum(self.effectifs)
        cumsum = np.concatenate(([0], cumsum))
        self.cumsum = cumsum
        self.DECOMPTE = 0
        


    def tirage(self, random_state, n_samples, n_samples_bootstrap):
        self.DECOMPTE += 1
        #if (self.DECOMPTE % 10) == 0:
        logging.debug("tirage N°%d"%self.DECOMPTE)
        random_instance = check_random_state(random_state)
        classes = random_instance.choice(
            self.range_classes,
            size=n_samples_bootstrap
        )
        v_func = np.vectorize(
            lambda n: random_instance.randint(self.cumsum[n],self.cumsum[n+1])
        )
        indices = v_func(classes)
        indices = self.indices_ordre[indices]
        #print("fin.", end="\r")
        logging.debug("fin.")
        return indices
    
    def __call__(self, random_state, n_samples, n_samples_bootstrap):
        return self.tirage(random_state, n_samples, n_samples_bootstrap)
    

def tirage_sans_remise(random_state, sampler, nb_par_classe):
    random_instance = check_random_state(random_state)
    cumsum = sampler.cumsum
    ars = []
    for c in range(sampler.nb_classes):
      ars.append(random_instance.choice(np.arange(cumsum[c], cumsum[c+1]),
                                        size = nb_par_classe,
                                        replace=False))
    ars = np.concatenate(ars)
    return sampler.indices_ordre[ars]
   
    
def compute_proximity(RF, X, output_leaves=False):
    #RF : La forêt
    #X  : Le ou les échantillons (numpy)
    n_samples, p = X.shape
    noeuds, idx = RF.decision_path(X)
    #print(type(noeuds))
    logging.info(str(type(noeuds)))
    ntot = idx[-1]
    idx = idx[:-1] #La dernière valeur est inutile
    idx = idx[::-1]
    F = dok_array((n_samples, ntot), dtype=np.int32)
    for i in tqdm.tqdm(range(n_samples), total=n_samples):
        roots = iter(idx)
        root = next(roots)
        Li = noeuds[i, :]
        Li = [_ for _ in zip(*Li.nonzero())]
        Li.sort()
        Li = Li[::-1]
        #La première valeur est forcément un numéro de feuille.
        _, nd = Li[0]
        F[i,nd] = 1
        Li = Li[1:]

        k = 0
        Nli = len(Li)
        while k < Nli:
            _, nd = Li[k]
            if nd == root:
                root = next(roots)
                k += 1
                _, nd = Li[k]
                F[i,nd] = 1
                if root == 0:
                    break
            k += 1
    if output_leaves:
        return F
    tF = F.transpose()
    prox = F.dot(tF)
    prox = prox / len(idx)
    return prox

def compute_proximity_2(RF, X, output_leaves=True):
    #RF : La forêt
    #X  : Le ou les échantillons (numpy)
    n_samples, p = X.shape
    noeuds, idx = RF.decision_path(X)
    #print(type(noeuds))
    logging.info(str(type(noeuds)))
    ntot = idx[-1]
    idx = idx[:-1] #La dernière valeur est inutile
    idx = idx[::-1]
    nb_trees = RF.n_estimators
    #F = dok_array((n_samples, ntot), dtype=np.int32)
    F = np.zeros((n_samples, nb_trees), dtype=np.int32)
    for i in tqdm.tqdm(range(n_samples), total=n_samples):
        roots = iter(idx)
        root = next(roots)
        Li = noeuds[i, :]
        Li = [_ for _ in zip(*Li.nonzero())]
        Li.sort()
        Li = Li[::-1]
        #La première valeur est forcément un numéro de feuille.
        _, nd = Li[0]
        #F[i,nd] = 1
        #Li = Li[1:]
        jj = nb_trees
        jj -=1
        F[i, jj] = nd

        k = 0
        Nli = len(Li)
        while k < Nli:
            _, nd = Li[k]
            if nd == root:
                root = next(roots)
                k += 1
                _, nd = Li[k]
                #F[i,nd] = 1
                jj -= 1
                F[i,jj] = nd
                if root == 0:
                    break
            k += 1
    if output_leaves:
        return F
    
    
def importer_dataset(nom_fich):
    datafr = pnd.read_feather(nom_fich)
    datafr["relation"].fillna("<no_rel>", inplace=True)
    relations = datafr.groupby(datafr["relation"]).size()
    relations = relations.sort_values(ascending=False)

    return datafr, relations


def filtrer_dataset(datafr, fonction):
    # fonction prend un nom de relation. Elle doit
    # retourner None pour éliminer les lignes
    # Elle retourne une autre relation Sinon
    elim = datafr["relation"].apply(
        lambda x: (fonction(x) is not None)
    )
    resu = datafr[elim]
    rel_filtered = resu["relation"].apply(fonction)

    resu = resu.copy()
    return resu, rel_filtered



def calculer_effectifs(y):
    y, dic_list = pnd.factorize(y)
    nb_classes = len(dic_list)
    effectifs = np.bincount(y, minlength=nb_classes)
    dic_list = dic_list.to_list()
    effectifs = effectifs.tolist()
    return dic_list, effectifs


def importer_et_filter_dataset(nom_fich):
    datafr = pnd.read_feather(nom_fich)
    datafr["relation"].fillna("<no_rel>", inplace=True)
    relations = datafr.groupby(datafr["relation"]).size()
    relations = relations.sort_values(ascending=False)
    # Ne retenir que les 24 premières classes
    retenues = relations.index[:24].to_list()
    retenues.append("<other>")
    rel_filtered = datafr["relation"].apply(
        lambda x: x if x in retenues else "<other>"
    )
    return datafr, rel_filtered

def importer_filtrer_et_decouper_dataset(nom_fich, random_state):
    datafr = pnd.read_feather(nom_fich)
    datafr["relation"].fillna("<no_rel>", inplace=True)
    relations = datafr.groupby(datafr["relation"]).size()
    relations = relations.sort_values(ascending=False)
    # Ne retenir que les 24 premières classes
    retenues = relations.index[:24].to_list()
    retenues.append("<other>")
    rel_filtered = datafr["relation"].apply(
        lambda x: x if x in retenues else "<other>"
    )

    X_train, X_tstval, y_train, y_tstval = train_test_split(datafr, rel_filtered, train_size=0.9, stratify=rel_filtered)

    return X_train, X_tstval, y_train, y_tstval
    


class Hparams():
    def __init__(self, **kwargs):
        self.dico = dict()
        for kw, val in kwargs.items():
            self.dico[kw] = val

    def __getattr__(self, kw):
        if not kw in self.dico:
            return None
        else:
            return self.dico[kw]

def calculer_foret(DF, rel_filtered, oob_score=True, hparams=Hparams()):
    dico = dict()
    if hparams.n_estimators == None:
        hparams.n_estimators = 8000
    dico["n_estimators"] = hparams.n_estimators
    dico["oob_score"] = oob_score
    if hparams.max_features != None:
        dico["max_features"] = hparams.max_features
    if hparams.min_samples_leaf != None:
        dico["min_samples_leaf"] = hparams.min_samples_leaf
    if hparams.max_samples == None:
        hparams.max_samples = 50000
    dico["max_samples"] = hparams.max_samples
    if hparams.max_depth != None:
        dico["max_depth"] = hparams.max_depth
    if hparams.variables_retenues != None:
        assert hparams.variables_retenues in ["toutes", "sans_token_sep"]
        if hparams.variables_retenues == "toutes":
            sl = slice(5,None) #slice du type [5:]
        else:
            sl = slice(5,5+2*144) #slice du type [5:5+2*144]
    else:
        sl = slice(5,None) #slice du type [5:]
    if hparams.random_state != None:
        dico["random_state"] = hparams.random_state
    dico["n_jobs"] = -1
    
    
    RandoFo = RandoFo_Custom_Bootstrap(**dico)

    chaine = " ; ".join("=".join([k,str(v)]) for k,v in hparams.dico.items())
    logging.info(chaine)

    XXX = DF.iloc[:,sl] # la variable sl est la slice sélectionnée plus haut.
    if hparams.features != None and type(hparams.features) is list:
        XXX = XXX.iloc[:, hparams.features]
    sampler = DS_sampler(XXX, rel_filtered)
    RandoFo.set_sampler(sampler)
    RandoFo.fit(sampler.X, sampler.y)
    return RandoFo


def interactive_plot(projections, yyy, nom_fichier_figure):

    def label_fn(row):
        """HTML printer over each example."""
        return "<br>".join([f"<b>{k}:</b> {v}" for k, v in row.items()])
    
    couleurs = [px.colors.qualitative.G10[y_%10] for y_ in yyy]
    symboles = ["x-thin-open", "circle-open", "diamond-open"]
    symboles = [symboles[y_ % 3] for y_ in yyy]

    plotly.offline.plot(
        {
            "data": [
                gr_ob.Scatter(
                    x=projections[:, 0],
                    y=projections[:, 1],
                    mode="markers",
                    marker={
                        "color": couleurs,
                        "size": 8,
                        "symbol": symboles
                    }
                )
            ],
            "layout": gr_ob.Layout(width=800, height=800, template="simple_white"),
        },
        filename=nom_fichier_figure
    )


def plot_confusion_matrix(labels, preds, label_names, imgfile=None, numeric=True):
    if numeric:
        confusion_norm = confusion_matrix(labels,
                                        preds, #.tolist(),
                                        labels=list(range(len(label_names))),
                                        normalize="true")
        confusion = confusion_matrix(labels,
                                    preds, #.tolist(),
                                    labels=list(range(len(label_names)))
                                    )
    else:
        confusion_norm = confusion_matrix(labels,
                                        preds, #.tolist(),
                                        labels=label_names,
                                        normalize="true")
        confusion = confusion_matrix(labels,
                                    preds, #.tolist(),
                                    labels=label_names
                                    )
    #print(confusion)

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
    if imgfile == None:
        plt.show()
    #elif hasattr(imgfile, "format_image"):
    #    #La classe HTML_IMAGE peut contenir son format de fichier
    #    fig.savefig(imgfile, format=imgfile.format_image)
    else:
        fig.savefig(imgfile)
    plt.close()
    return confusion


def plot_importance(F, imgfile=None):
    
    assert F.shape == (4*144,) or F.shape == (2*144,) or F.shape == (144,)
    
    mini, maxi = np.min(F), np.max(F)
    
    if F.shape == (4*144,):
        fig, axes = plt.subplots(2,2,figsize=(16, 14))
        Fsc = F[:144].reshape((12,12))
        Fcs = F[144:288].reshape((12,12))
        Fssep = F[288:288+144].reshape((12,12))
        Fcsep = F[288+144:].reshape((12,12))
        data = [[Fsc, Fcs],[Fssep, Fcsep]]
        titres = [["source -> cible", "cible -> source"],["source -> sep", "cible -> sep"]]
        N = 2
        NN = 2
    elif F.shape == (2*144,):
        fig, axes = plt.subplots(1,2,figsize=(16, 7))
        Fsc = F[:144].reshape((12,12))
        Fcs = F[144:288].reshape((12,12))
        axes = axes.reshape((1,2))
        data = [[Fsc, Fcs]]
        titres = [["source -> cible", "cible -> source"]]
        N = 1
        NN = 2
    elif F.shape == (144,):
        fig, axes = plt.subplots(figsize=(7, 7))
        Ftt = F[:].reshape((12,12))
        axes = np.array([[axes]])
        titres = [["token -> previous_token"]]
        data = [[Ftt]]
        N = 1
        NN = 1
    else:
        plt.close()
        return
    
    for I in range(N):
        for J in range(NN):
            annot = data[I][J]
            mini = np.min(annot)
            maxi = np.max(annot)
            annot = (annot - mini)/(maxi-mini)
            sbn.heatmap(
                data[I][J],
                vmin = mini,
                vmax = maxi,
                annot=annot,
                cbar=False,
                fmt=".02f",
                xticklabels=["H%d"%H for H in range(12)],
                yticklabels = ["L%d"%L for L in range(12)],
                cmap="viridis",
                ax=axes[I,J]
            )
            axes[I,J].set_xlabel("Head idx")
            axes[I,J].set_ylabel("Layer idx")
            axes[I,J].set_title(titres[I][J])
    
    if imgfile == None:
        plt.show()
    #elif hasattr(imgfile, "format_image"):
    #    #La classe HTML_IMAGE peut contenir son format de fichier
    #    fig.savefig(imgfile, format=imgfile.format_image)
    else:
        fig.savefig(imgfile)
    plt.close()
    


def calcul_1():
    #print("Chargement du dataset", file=sys.stderr)
    logging.info("Chargement du dataset")
    DF, rel_filtered = importer_et_filter_dataset("./dataframe_QscalK.fth")
    #DF, rel_filtered = importer_et_filter_dataset("./dataframe_attention.fth")
    #("Lancement du calcul de la forêt", file=sys.stderr)
    logging.info("Lancement du calcul de la forêt")
    RandoFo = calculer_foret(DF, rel_filtered)
    #print("Calcul terminé. Sauvegarde", file=sys.stderr)
    logging.info("Calcul terminé. Sauvegarde")
    joblib.dump(RandoFo, "./RandoFo_QK_12_8000_50000.joblib")
    #print("Terminé.", file=sys.stderr)
    logging.info("Terminé")

def calcul_1_5():
    logging.info("Chargement du dataset")
    #DF, rel_filtered = importer_et_filter_dataset("./dataframe_QscalK.fth")
    DF, rel_filtered = importer_et_filter_dataset("./dataframe_attention.fth")
    logging.info("Lancement du calcul de la forêt")
    RandoFo = calculer_foret(DF, rel_filtered)
    logging.info("Exactitude OOB : %s"%str(RandoFo.oob_score_))
    logging.info("Calcul du résumé.")
    resume = RandoFo_Summary(RandoFo)
    logging.info("Calcul terminé. Sauvegarde")
    joblib.dump(resume, "./RFresume_att_12_8000_50000.joblib")
    logging.info("Terminé")


def summarize_QK_RF():
    logging.info("Chargement de la forêt aléatoire")
    RandoFo = joblib.load("./RandoFo_QK_12_8000_50000.joblib")
    logging.info("Exactitude OOB : %s"%str(RandoFo.oob_score_))
    logging.info("Calcul du résumé.")
    resume = RandoFo_Summary(RandoFo)
    logging.info("Calcul terminé. Sauvegarde")
    joblib.dump(resume, "./RFresume_QK_12_8000_50000.joblib")
    logging.info("Terminé")

def calcul_2():
    logging.info("Chargement de la forêt aléatoire")
    RandoFo = joblib.load("./RandoFo_QK_12_8000_50000.joblib")
    logging.info("Exactitude OOB : %s"%str(RandoFo.oob_score_))
    logging.info("Tirage d’un échantillon de 1000 individus")
    sampler = RandoFo._custom_sample
    #idx_test = sampler.tirage(56, sampler.n_samples, 1000)
    idx_test = tirage_sans_remise(56, sampler, 40)
    X_test = sampler.X[idx_test]
    y_test = sampler.y[idx_test]
    logging.info("Calcul de la matrice de proximité")
    prox = compute_proximity(RandoFo, X_test)
    logging.info("prox.shape : %s"%str(prox.shape))
    prox = prox.todense()
    logging.info("Sauvegarde de la matrice de proximité")
    joblib.dump(prox, "./figures/prox_QK_12_8000_50000.joblib")
    joblib.dump(y_test, "./figures/y_test_QK_12_8000_50000.joblib")
    logging.info("Tracé des figures.")
    Truc = prox.copy()
    Truc = Truc.ravel()
    Truc.sort()
    fig, axes = plt.subplots()
    axes.plot(np.arange(len(Truc)), Truc)
    plt.savefig("./figures/valeurs_prox.png")
    Truc = prox.copy()
    for i in range(Truc.shape[0]):
        Truc[i,i] = 0
    fig, axes = plt.subplots()
    axes.imshow(prox)
    plt.savefig("./figures/matrix_prox.png")
    non_nul = np.sum(Truc != 0)
    message_info = "Nombre de similatités non-nulles : %d. Proportion : %f"%(non_nul, non_nul/(Truc.shape[0]*Truc.shape[1]))
    logging.info(message_info)
    logging.info("Terminé")

def calcul_3():
    logging.info("Chargement")
    prox = joblib.load("./figures/prox_QK_12_8000_50000.joblib")
    y_test = joblib.load("./figures/y_test_QK_12_8000_50000.joblib")
    logging.info("Tracé de la matrice de proximité")
    fig, axes = plt.subplots()
    axes.imshow(prox != 0)
    plt.savefig("./figures/matrix_prox.png")
    logging.info("Calcul du tSNE")
    t_sne = TSNE(n_components=2,
        perplexity=20,
        metric="precomputed",
        init="random",
        verbose=1,
        learning_rate="auto").fit_transform(1.-prox)
    logging.info("Tracé du tSNE")
    interactive_plot(t_sne, y_test, "./figures/clusters_tSNE.html")
    logging.info("Terminé")


def batch_essai_reproductibilite():
    nom_rapport = "./essai_reproductibilite.html"
    nom_dataset = "./dataframe_attention.fth"
    train_size = 0.9
    random_state = 152331
    for _ in range(2):
        logging.info("Chargement du dataset")
        datafr, relations = importer_dataset(nom_dataset)
        dic_list, effectifs = calculer_effectifs(datafr["relation"])
        
        with HTML_REPORT(nom_rapport) as R:
            R.ligne()
            R.titre("Dataset : %s"%nom_dataset)
            R.texte("Effectifs avant filtrage :")
            R.table(relations=dic_list, effectifs=effectifs)
        elim = [x for x in relations.index if x.startswith(":snt")]
        elim.extend([x for x in relations.index if x.startswith(":op")])
        elim.append(":polarity")
        filtrer = lambda x: None if x in elim else ("<other>" if relations[x] < 1000 else x)
        datafr, rels_f = filtrer_dataset(datafr, filtrer)

        X_train, X_tstval, y_train, y_tstval = train_test_split(
            datafr,
            rels_f,
            train_size=train_size,
            random_state = random_state,
            stratify=rels_f)
        
        
        with HTML_REPORT(nom_rapport) as R:
            R.titre("filtrage", 2)
            R.table(avant=relations.index.to_list(), apres=[filtrer(x) for x in relations.index])
            R.texte("Effectifs après filtrage :")
            dic_list, effectifs = calculer_effectifs(rels_f)
            R.table(relations=dic_list, effectifs=effectifs)
            R.table(Random_state_pour_sep_train_test = random_state,
                    proportion_dans_train=train_size,
                    colonnes=False)
            
        sl = slice(5,5+2*144)
        hparams = Hparams(
                        n_estimators=200,
                        max_features=12,
                        max_samples=5000,
                        variables_retenues = "sans_token_sep",
                        random_state = 51223
                    )
        logging.info("Calcul de la forêt")
        RF = calculer_foret(X_train, y_train, oob_score=False, hparams=hparams)
        logging.info("Fin du calcul de la forêt")
        with HTML_REPORT(nom_rapport) as R:
            R.titre("Paramètres pour la forêt",2)
            R.table(
                colonnes=False,
                **hparams.dico
            )

        logging.info("Calcul des perfs")
        dic_list = RF._custom_sample.dic_list
        dic_dic = {k:i for i,k in enumerate(dic_list)}
        realite = y_tstval.apply(lambda x: dic_dic[x])
        pred = RF.predict(X_tstval.iloc[:,sl])
        accuracy = accuracy_score(realite, pred)
        importance = RF.feature_importances_
        with HTML_REPORT(nom_rapport) as R:
            R.titre("Accuracy : %f"%accuracy, 2)
            R.titre("Matrice de confusion", 2)
            with R.new_img("png") as IMG:
                confusion = plot_confusion_matrix(realite, pred, dic_list, IMG)
            R.texte("confusion au format python :")
            R.texte(repr(confusion))
            R.titre("Importance des caractères", 2)
            with R.new_img("png") as IMG:
                plot_importance(importance, imgfile=IMG)
            mini, maxi = np.min(importance[:144]), np.max(importance[:144])
            R.texte("mini : %f, maxi : %f"%(mini, maxi))
            R.ligne()
        


def batch_quatre_meilleurs():
    nom_rapport = "./quatre_meilleurs.html"
    noms_dataset = ["./dataframe_attention.fth", "./dataframe_QscalK.fth"]
    random_state = 152331
    train_size = 0.9
    try:
        for nom_dataset in noms_dataset:
            logging.info("Chargement du dataset")
            datafr, relations = importer_dataset(nom_dataset)
            dic_list, effectifs = calculer_effectifs(datafr["relation"])
            
            with HTML_REPORT(nom_rapport) as R:
                R.ligne()
                R.titre("Dataset : %s"%nom_dataset)
                R.texte("Effectifs avant filtrage :")
                R.table(relations=dic_list, effectifs=effectifs)

            logging.info("Dataset : %s"%nom_dataset)
            elim = [x for x in relations.index if x.startswith(":snt")]
            elim.extend([x for x in relations.index if x.startswith(":op")])
            elim.append(":polarity")
            filtrer = lambda x: None if x in elim else ("<other>" if relations[x] < 1000 else x)
            datafr, rels_f = filtrer_dataset(datafr, filtrer)

            X_train, X_tstval, y_train, y_tstval = train_test_split(
                datafr,
                rels_f,
                train_size=train_size,
                random_state = random_state,
                stratify=rels_f)

            with HTML_REPORT(nom_rapport) as R:
                R.titre("filtrage", 2)
                R.table(avant=relations.index.to_list(), apres=[filtrer(x) for x in relations.index])
                R.texte("Effectifs après filtrage :")
                dic_list, effectifs = calculer_effectifs(rels_f)
                R.table(relations=dic_list, effectifs=effectifs)
                R.table(Random_state_pour_sep_train_test = random_state,
                        proportion_dans_train=train_size,
                        colonnes=False)
            for varbls in ["sans_token_sep", "toutes"]:
                if varbls == "toutes":
                    sl = slice(5,None) #slice du type [5:]
                else:
                    sl = slice(5,5+2*144) #slice du type [5:5+2*144]
                if varbls == "sans_token_sep" and nom_dataset == "./dataframe_attention.fth":
                    pass
                    #Sauter : déjà fait.
                else:
                    hparams = Hparams(
                        n_estimators=8000,
                        max_features=100,
                        max_samples=50000,
                        variables_retenues = varbls,
                        random_state = 51223
                    )
                    logging.info("Calcul de la forêt")
                    RF = calculer_foret(X_train, y_train, oob_score=False, hparams=hparams)
                    logging.info("Fin du calcul de la forêt")

                    with HTML_REPORT(nom_rapport) as R:
                        R.titre("Paramètres pour la forêt",2)
                        R.table(
                            colonnes=False,
                            **hparams.dico
                        )

                    logging.info("Calcul des perfs")
                    dic_list = RF._custom_sample.dic_list
                    dic_dic = {k:i for i,k in enumerate(dic_list)}
                    realite = y_tstval.apply(lambda x: dic_dic[x])
                    pred = RF.predict(X_tstval.iloc[:,sl])
                    accuracy = accuracy_score(realite, pred)
                    importance = RF.feature_importances_
                    with HTML_REPORT(nom_rapport) as R:
                        R.titre("Accuracy : %f"%accuracy, 2)
                        R.titre("Matrice de confusion", 2)
                        with R.new_img("png") as IMG:
                            confusion = plot_confusion_matrix(realite, pred, dic_list, IMG)
                        R.titre("confusion au format python :", 3)
                        R.texte(repr(confusion))
                        R.titre("Importance des caractères", 2)
                        with R.new_img("png") as IMG:
                            plot_importance(importance, imgfile=IMG)
                        R.titre("Importance au format python :", 3)
                        R.texte(repr(importance))
                        R.ligne()

    except Exception as e:
        chaine = traceback.format_exc()
        logging.error(chaine)
        with HTML_REPORT(nom_rapport) as R:
            R.ligne()
            R.titre("Une erreur est survenue !")
            R.texte(chaine)


def batch():
    nom_rapport = "./rapport_randofo.html"
    noms_dataset = ["./dataframe_attention.fth", "./dataframe_QscalK.fth"]
    random_state = 152331
    liste_n_estimators = [500, 1000, 2000, 5000, 8000]
    liste_max_features = [None, 12, 25, 36, 50, 100]
    liste_max_samples = [5000, 10000, 25000, 50000]
    train_size = 0.9
    try:
        for nom_dataset in noms_dataset:
            logging.info("Chargement du dataset")
            datafr, relations = importer_dataset(nom_dataset)
            dic_list, effectifs = calculer_effectifs(datafr["relation"])
            
            with HTML_REPORT(nom_rapport) as R:
                R.ligne()
                R.titre("Dataset : %s"%nom_dataset)
                R.texte("Effectifs avant filtrage :")
                R.table(relations=dic_list, effectifs=effectifs)

            logging.info("Dataset : %s"%nom_dataset)
            elim = [x for x in relations.index if x.startswith(":snt")]
            elim.extend([x for x in relations.index if x.startswith(":op")])
            elim.append(":polarity")
            filtrer = lambda x: None if x in elim else ("<other>" if relations[x] < 1000 else x)
            datafr, rels_f = filtrer_dataset(datafr, filtrer)

            X_train, X_tstval, y_train, y_tstval = train_test_split(
                datafr,
                rels_f,
                train_size=train_size,
                random_state = random_state,
                stratify=rels_f)

            with HTML_REPORT(nom_rapport) as R:
                R.titre("filtrage", 2)
                R.table(avant=relations.index.to_list(), apres=[filtrer(x) for x in relations.index])
                R.texte("Effectifs après filtrage :")
                dic_list, effectifs = calculer_effectifs(rels_f)
                R.table(relations=dic_list, effectifs=effectifs)
                R.table(Random_state_pour_sep_train_test = random_state,
                        proportion_dans_train=train_size,
                        colonnes=False)

            for varbls in ["sans_token_sep", "toutes"]:
                if varbls == "toutes":
                    sl = slice(5,None) #slice du type [5:]
                else:
                    sl = slice(5,5+2*144) #slice du type [5:5+2*144]
                for _ in range(20):
                    hparams = Hparams(
                        n_estimators=random.choice(liste_n_estimators),
                        max_features=random.choice(liste_max_features),
                        max_samples=random.choice(liste_max_samples),
                        variables_retenues = varbls
                    )
                    logging.info("Calcul de la forêt")
                    RF = calculer_foret(X_train, y_train, oob_score=False, hparams=hparams)
                    logging.info("Fin du calcul de la forêt")

                    with HTML_REPORT(nom_rapport) as R:
                        R.titre("Paramètres pour la forêt",2)
                        R.table(
                            colonnes=False,
                            **hparams.dico
                        )

                    logging.info("Calcul des perfs")
                    dic_list = RF._custom_sample.dic_list
                    dic_dic = {k:i for i,k in enumerate(dic_list)}
                    realite = y_tstval.apply(lambda x: dic_dic[x])
                    pred = RF.predict(X_tstval.iloc[:,sl])
                    accuracy = accuracy_score(realite, pred)
                    with HTML_REPORT(nom_rapport) as R:
                        R.titre("Accuracy : %f"%accuracy, 2)
                        R.titre("Matrice de confusion", 2)
                        with R.new_img("png") as IMG:
                            confusion = plot_confusion_matrix(realite, pred, dic_list, IMG)
                        R.texte("confusion au format python :")
                        R.texte(repr(confusion))
                        R.ligne()

        
    except Exception as e:
        chaine = traceback.format_exc()
        logging.error(chaine)
        with HTML_REPORT(nom_rapport) as R:
            R.ligne()
            R.titre("Une erreur est survenue !")
            R.texte(chaine)



if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s :: %(levelname)s :: %(message)s',
        filename='batch_calcul.log',
        encoding='utf-8',
        level=logging.INFO
    )
    #calcul_1_5()
    #summarize_QK_RF()

    #batch()
    #batch_essai_reproductibilite()
    #batch_quatre_meilleurs()
    batch_MLP()
    if False:
        realite = [0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,1,2,3,4,5,6]
        pred =    [0,2,2,3,4,5,0,0,1,2,3,1,5,6,0,1,2,3,4,5,6,0,5,2,3,2,5,6]
        lbls = ["zero", "un", "deux", "trois", "quatre", "cinq", "six"]

        #plot_confusion_matrix(realite, pred, lbls, "png.png")
        
        with HTML_REPORT("./rapport_essai.html") as R:
            R.titre("Matrice de confusion")
            with R.new_img("png") as IMG:
                plot_confusion_matrix(realite, pred, lbls, IMG)
            R.texte("Voilà.")

    


