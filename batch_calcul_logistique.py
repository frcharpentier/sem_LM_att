
import traceback
import joblib

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


import logging
from report_generator import HTML_REPORT

#from batch_calcul_BERT import RandoFo_Summary, RandoFo_Custom_Bootstrap, DS_sampler
from batch_calcul_BERT import Hparams, plot_confusion_matrix, plot_importance
from batch_calcul_BERT import importer_dataset, calculer_effectifs, filtrer_dataset
from batch_calcul_BERT import calculer_foret

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler


def batch_Lgstq():
    nom_rapport = "./classif_Logistiq.html"
    
    noms_dataset = ["./dataframe_QscalK.fth"] #, "./dataframe_QscalK_gpt2.fth", "./dataframe_QscalK_gpt2_CSSC.fth"]
    random_state = 152331
    #RS_MLP = 152331
    RS_MLP = 425
    RS_oversampling = 8236
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1

    dual = False
    tol=1.0e-4
    valC = 1.0
    fit_intercept = True
    cw = "balanced"
    solver = "saga" #"lbfgs"
    max_iter=150
    multi_class = "multinomial"
    

    try:
        with HTML_REPORT(nom_rapport) as R:
            for nom_dataset in noms_dataset:
                logging.info("Chargement du dataset %s"%(nom_dataset))
                datafr, relations = importer_dataset(nom_dataset)
                dic_list, effectifs = calculer_effectifs(datafr["relation"])

                R.ligne()
                R.titre("Dataset : %s"%nom_dataset)
                R.texte("Effectifs avant filtrage :")
                R.table(relations=dic_list, effectifs=effectifs)
                R.flush()

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
                
                X_val, X_tst, y_val, y_tst = train_test_split(
                    X_tstval,
                    y_tstval,
                    train_size = val_size / (val_size + test_size),
                    random_state = random_state,
                    stratify=y_tstval)
                
                R.titre("filtrage", 2)
                R.table(avant=relations.index.to_list(), apres=[filtrer(x) for x in relations.index])
                R.texte("Effectifs après filtrage :")
                dic_list, effectifs = calculer_effectifs(rels_f)
                R.table(relations=dic_list, effectifs=effectifs)
                R.table(Random_state_pour_sep_train_test = random_state,
                        proportion_dans_train=train_size,
                        Random_state_pour_oversampling = RS_oversampling,
                        colonnes=False)
                R.flush()
                    
                for varbls in ["sans_token_sep"]: #["sans_token_sep", "toutes"]:
                    if varbls == "toutes":
                        sl = slice(5,None) #slice du type [5:]
                    else:
                        sl = slice(5,5+2*144) #slice du type [5:5+2*144]
                    modele = R.skl(LogisticRegression)(
                        dual=dual,
                        tol=tol,
                        C=valC,
                        fit_intercept=fit_intercept,
                        class_weight = cw,
                        random_state=RS_MLP,
                        solver=solver,
                        max_iter=max_iter,
                        multi_class = multi_class,
                        verbose=3
                    )
                    
                    logging.info("Calcul de la régression (logistique)")
                    y_tr = y_train.to_numpy()
                    x_tr = X_train.iloc[:, sl].to_numpy()
                    modele.fit(x_tr, y_tr)
                    #mlp.partial_fit(x_tr, y_tr, np.unique(y_tr))
                    logging.info("fin du calcul.")
                    logging.info("Calcul des perfs.")
                    predictions = modele.predict(X_val.iloc[:, sl].to_numpy())
                    accuracy = accuracy_score(y_val, predictions)
                    bal_accuracy = balanced_accuracy_score(y_val, predictions)
                    logging.info("Accuracy : %f, bal_accuracy : %f"%(accuracy, bal_accuracy))
                    dtype1 = predictions.dtype
                    dtype0 = y_val.dtype
                    logging.info("dtype %s %s"%(repr(dtype0),repr(dtype1)))
                    logging.info("mlp.classes_ : %s"%repr(modele.classes_))
                    logging.info("predictions %s"%(repr(predictions[:20])))
                    
                    R.titre("Accuracy : %f, balanced accuracy : %f"%(accuracy, bal_accuracy), 2)
                    R.titre("Matrice de confusion", 2)
                    with R.new_img_with_format("svg") as IMG:
                        confusion = plot_confusion_matrix(y_val.to_numpy(), predictions, dic_list, IMG.fullname, numeric=False)
                    R.titre("confusion au format python :", 3)
                    R.texte(repr(confusion))
                    R.ligne()
                    R.flush()
                    logging.info("Sauvegarde du modèle")
                    with R.new_ressource() as RES:
                        joblib.dump(modele, RES.fullname)

    except Exception as e:
        chaine = traceback.format_exc()
        logging.error(chaine)
        with HTML_REPORT(nom_rapport) as R:
            R.ligne()
            R.titre("Une erreur est survenue !")
            R.texte(chaine)






if __name__=="__main__":
    logging.basicConfig(
        format='%(asctime)s :: %(levelname)s :: %(message)s',
        filename='batch_calcul.log',
        encoding='utf-8',
        level=logging.INFO
    )
    #essai_oversampling()
    batch_Lgstq()