import sys
import traceback
import logging
from report_generator import HTML_REPORT

import pandas as pnd
from scipy import stats
from IPython.display import display
import matplotlib.pyplot as plt
import json
import tqdm

import sklearn
print(sklearn.__version__)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

DF_att = pnd.read_feather("./dataframe_attention.fth")
DF_att["relation"].fillna("<no_rel>", inplace=True)
DF_QK = pnd.read_feather("./dataframe_QscalK.fth")
DF_QK["relation"].fillna("<no_rel>", inplace=True)

DF_att = DF_att.iloc[:, :5+144]
DF_att.rename(columns = (lambda x: x[:-3] if (x.endswith("_SC") or x.endswith("_TT")) else x), inplace=True)
DF_QK = DF_QK.iloc[:, :5+144]
DF_QK.rename(columns = (lambda x: x[:-3] if (x.endswith("_SC") or x.endswith("_TT")) else x), inplace=True)

relations_att = DF_att.groupby(DF_att["relation"]).size()
relations_att = relations_att.sort_values(ascending=False)

relations_QK = DF_QK.groupby(DF_att["relation"]).size()
relations_QK = relations_QK.sort_values(ascending=False)
display(relations_att)
display(relations_QK)


groups = {
    "att" : DF_att.groupby(DF_att["relation"]),
    "QK"  : DF_QK.groupby(DF_QK["relation"])
}
colonnes = DF_att.columns
relations = DF_att["relation"].unique()

def extract_head_rel(attQK, head, relation):
    assert attQK in groups
    if not head.startswith("att_"):
        head = "att_" + head + "_TT"
    assert head in colonnes
    columns = [c for c in colonnes[:4]] + [head]
    dfout = groups[attQK].get_group(relation)[columns].copy()
    return dfout


class ACP:
    def __init__(self):
        self.sc = StandardScaler()
        self.acp = PCA()
        
    def fit(self, dataf):
        self.dataf = dataf
        self.Z = self.sc.fit_transform(dataf[colonnes[5:]])
        #self.Z = dataf[colonnes[5:]].to_numpy()
        self.nobs = self.Z.shape[0]
        self.coord = self.acp.fit_transform(self.Z)
        self.ncomp = self.acp.n_components_
        self.eigval = ((self.nobs-1)/self.nobs)*self.acp.explained_variance_
        self.evr = self.acp.explained_variance_ratio_
        self.mini = np.min(self.Z)
        self.maxi = np.max(self.Z)
        
    def scree_plot(self, p):
        plt.plot(np.arange(1, p+1), self.eigval[:p])
        plt.title("Scree plot")
        plt.ylabel("Valeurs propres")
        plt.xlabel("Numéro de VP")
        plt.show()
    
    def cumul_variance(self, p):
        cumul = np.cumsum(self.evr)[:p]
        print(cumul)
        plt.plot(np.arange(1,p+1), cumul)
        plt.title("Variance expliquée")
        plt.ylabel("Pourcentage de la variance totale")
        plt.xlabel("Numéro de VP")
        plt.show()
        
    def plot_individuals(self, relations, noms="pipo", kde=False, contour=True, scatter=False, bw_method=None):
        print("plot_individuals")
        if type(relations) is list:
            assert len(relations) == 2 and all(type(X) is str for X in relations)
        else:
            relations = [relations]
            
        coord = self.coord[:, 0:4]  # On retient quatre composantes principales
        
        minima = np.min(coord, axis=0).tolist()
        maxima = np.max(coord, axis=0).tolist()
        
        liste_ij = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        dico_ij = dict()
        
        for ij in liste_ij:
            fig, axes = plt.subplots()
            dico_ij[ij] = (fig, axes)
        
        
        couleurs = ["red", "green"]
        
        for relation, couleur in zip(relations, couleurs):
        
            crd_filtr = coord[self.dataf["relation"]==relation, :]
            print(crd_filtr.shape)

            # Il y a trop d’individus. On va calculer une KDE et afficher l’intensité
            # en chaque point à la place.

            for (PC1, PC2), (FIG, AX) in dico_ij.items():
                xmin, ymin = minima[PC1], minima[PC2]
                xmax, ymax = maxima[PC1], maxima[PC2]
                print("PC%d : [%f , %f] PC%d : [%f , %f]"%(1+PC1, xmin, xmax, 1+PC2, ymin, ymax))

                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                kernel = stats.gaussian_kde(crd_filtr[:, [PC1, PC2]].T, bw_method=bw_method)
                f = np.reshape(kernel(positions), xx.shape)
                AX.set_xlabel("PC%d"%(1+PC1))
                AX.set_ylabel("PC%d"%(1+PC2))
                AX.set_xlim([xmin, xmax])
                AX.set_ylim([ymin, ymax])

                if kde:
                    print("kde")
                    AX.imshow(np.rot90(f),
                            #cmap = plt.cm.gist_earth_r,
                            extent = [xmin, xmax, ymin, ymax])
                    #AX.imshow(np.rot90(f))
                else:
                    print("sans kde")


                if contour:
                    print("contour")
                    cset = AX.contour(xx, yy, f, colors=couleur, )
                    # Label plot
                    #axes.clabel(cset, inline=1, fontsize=10)
                else:
                    print("sans contour")

                if scatter:
                    print("scatter")
                    AX.plot(crd_filtr[:,PC1], crd_filtr[:,PC2],
                        'k.',
                        markersize=2,
                        c=couleur)
                else:
                    print("sans scatter")
                
                nom_fichier = "%s_PC%d_PC%d.eps"%(noms, PC1+1, PC2+1)
                FIG.savefig(nom_fichier)
            
        
        #plt.show()
        
    def cos2(self):
        #Contribution des individus dans l’inertie totale :
        self.di = np.sum(self.Z**2, axis=1)
        #cos² :
        self.cos2 = (self.coord**2)/di.reshape((-1,1)) # calcul qui tire parti de la distributivité.
        #print(np.sum(cos2, axis=1)[:15])
        
    def prepare_plot_variables(self):
        eigvect = self.acp.components_
        sqrt_eigval = np.sqrt(self.eigval)
        
        #corrélation des variables avec les axes :
        self.corvar = np.zeros((self.ncomp, self.ncomp))
        for k in range(self.ncomp):
            self.corvar[:,k] = eigvect[k,:] * sqrt_eigval[k]
            
        
    def plot_variables(self, comp1, comp2):
        couleurs = ["#a6cee3", "#1f78b4", "#b2df8a",
                    "#33a02c", "#fb9a99", "#e31a1c",
                    "#fdbf6f", "#ff7f00", "#cab2d6",
                    "#6a3d9a", "#ffff99", "#b15928"]
        #Tracé du cercle de corrélation
        fig, axes = plt.subplots(figsize=(15,15))
        axes.set_xlim(-1,1)
        axes.set_ylim(-1,1)
        
        #affichage des étiquettes mentionnant les têtes d’atention :
        for j in range(self.ncomp):
            nomcol = colonnes[5+j][4:-3] # nom de la colonne, amputé du préfixe et du suffixe ("att_" et "_TT")
            num_couche = int(nomcol[1:nomcol.index("_")])
            plt.annotate(nomcol,
                         (self.corvar[j,comp1], self.corvar[j,comp2]),
                         color = couleurs[num_couche])
        
        #ajout des axes
        plt.plot([-1,1], [0,0], color="silver", linestyle="-", linewidth=1)
        plt.plot([0,0], [-1,1], color="silver", linestyle="-", linewidth=1)
        
        #dessiner un cercle
        circulus = plt.Circle((0,0), 1, color="blue", fill=False)
        axes.add_artist(circulus)
        
        plt.show()
        
    def export_json(self, ncomp, fichier_out):
        liste = self.coord[:, 0:ncomp].tolist()
        liste = [L + [self.dataf.iloc[i]["relation"]] for i, L in enumerate(liste)]
                
        with open(fichier_out, "w", encoding="utf-8") as F:
            json.dump(liste, F)

monACP = ACP()
monACP.fit(DF_QK)

monACP.plot_individuals(["<no_rel>", ":ARG0"], "./ACP_RoBERTa/norel_ARG0")
monACP.plot_individuals([":ARG0", ":ARG1"], "./ACP_RoBERTa/ARG0_ARG1")
monACP.plot_individuals([":condition", ":location"], "./ACP_RoBERTa/condition_location")
monACP.plot_individuals([":location",":time"], "./ACP_RoBERTa/location_time")
monACP.plot_individuals([":polarity", ":mod"], "./ACP_RoBERTa/polarity_mod")
monACP.plot_individuals([":quant", ":location"], "./ACP_RoBERTa/quant_location")
monACP.plot_individuals([":mod", ":poss"], "./ACP_RoBERTa/mod_poss")
monACP.plot_individuals([":ARG1", ":ARG2"], "./ACP_RoBERTa/ARG1_ARG2")



