from amr_utils.amr_readers import AMR_Reader
import os
import json
import random



class AMRSuivant(Exception):
    pass


def faire_json_paire_mots(fichier_produit):
    fichier_alignements = "./leamr/data-release/alignments/ldc+little_prince.subgraph_alignments.json"
    amr_rep = "../AMR_de_chez_LDC/LDC_2020_T02/data/alignments/unsplit"
    fichiers_amr = [os.path.abspath(os.path.join(amr_rep, f)) for f in os.listdir(amr_rep)]

    reader = AMR_Reader()
    amr_liste = []
    amr_dict = dict()

    for amrfile in fichiers_amr:
        #print(amrfile)
        listeG = reader.load(amrfile)
        amr_liste.extend(listeG)
        for graphe in listeG:
            amr_dict[graphe.id] = graphe

    #monamr = amr_dict["DF-199-192821-670_2956.4"]

    print("%d graphes AMR au total."%len(amr_liste))
    alignements = reader.load_alignments_from_json(fichier_alignements, amr_liste)
    phrases = []
    for idSNT, listAlig in alignements.items():
        try:
            if idSNT in amr_dict:
                graphe = amr_dict[idSNT]
                snt = None
                toks = None
                if "snt" in graphe.metadata:
                    snt = graphe.metadata["snt"]
                if hasattr(graphe, "tokens"):
                    toks = graphe.tokens
                    #print(type(toks))
                if snt == None:
                    snt = " ".join(toks)
                #print(idSNT, snt)
            else:
                continue
            paires = [(a.tokens[0], a.nodes[0]) for a in listAlig if(len(a.tokens) == 1 and len(a.nodes) == 1)]
            try:
                dicNoeuds = {graphe.isi_node_mapping[noeud] : (tok, toks[tok]) for tok, noeud in paires}
            except KeyError:
                        print("AMR suivant")
                        raise AMRSuivant
            
            phrase = False
            setAretes = set()
            for s, r, t in graphe.edges:
                if s <= t:
                    setAretes.add((s,t))
                else:
                    setAretes.add((t,s))
                if s in dicNoeuds and t in dicNoeuds:
                    numtok1, tok1 = dicNoeuds[s]
                    numtok2, tok2 = dicNoeuds[t]
                    concept1 = graphe.nodes[s]
                    concept2 = graphe.nodes[t]
                    if r.endswith('-of') and r not in [':consist-of', ':prep-out-of', ':prep-on-behalf-of']:
                        arete = (numtok2, tok2, concept2, r[:-3], numtok1, tok1, concept1)
                    elif r == ":domain":
                        #ERREUR ICI !!
                        #arete = (numtok2, tok2, concept2, r[:-3], numtok1, tok1, concept1)
                        #CORRECTION :
                        arete = (numtok2, tok2, concept2, ":mod", numtok1, tok1, concept1)
                        
                    else:
                        arete = (numtok1, tok1, concept1, r, numtok2, tok2, concept2)
                    if not phrase:
                        phrase = {"snt" : snt, "aretes" : [arete]}
                    else:
                        phrase["aretes"].append(arete)

            #Ajoutons des paires de mots sans aucune arête
            listeNoeuds = [x for x in dicNoeuds]
            Nmax = (len(graphe.relations)+5)//10
            # On calcule un nombre de paires vides égale à (environ) 10% du nombre de relations du graphe
            for _ in range(Nmax): 
                trouve = False
                random.shuffle(listeNoeuds)
                for s in listeNoeuds:
                    for t in listeNoeuds:
                        if t == s:
                            continue
                        if s <= t:
                            S,T = s,t
                        else:
                            S,T = t,s
                        if not (S,T) in setAretes:
                            numtok1, tok1 = dicNoeuds[s]
                            numtok2, tok2 = dicNoeuds[t]
                            concept1 = graphe.nodes[s]
                            concept2 = graphe.nodes[t]
                            arete = (numtok1, tok1, concept1, None, numtok2, tok2, concept2)
                            # Remarque : On a mis None pour la relation, pour indiquer que c’est une paire vide.
                            if not phrase:
                                phrase = {"snt": snt, "aretes" : [arete]}
                                trouve = True
                                break
                            elif not arete in phrase["aretes"]:
                                phrase["aretes"].append(arete)
                                trouve = True
                                break #Sortir de la boucle for t
                    if trouve:
                        break #Sortir de la boucle for s



            if phrase:
                phrases.append(phrase)

        except AMRSuivant:
            continue

    with open(fichier_produit, "w", encoding="UTF-8") as fout:
        json.dump(phrases, fout, indent=4)
    print("Terminé.") 

def aligner_tokenizers(phrase, tokenizer, tok_idxs):
    # tokenizer est un tokenizer de transformer (genre roberta, bart ou T5)
    # tok_idxs est une liste de paires (numero de mot, mot)
    # La fonction tokenize la phrase avec le tokenizer donné
    # et vérifie que le token numéro tant correspond bien au mot donné,
    # pour chacune des paires. S’il y a un décalage, elle le renvoie
    # Si un mot est scindé en plusieurs tokens, il est éliminé de la liste.
    indices = [i for i in range(len(tok_idxs))]
    indices.sort(key = lambda i: tok_idxs[i][0])
    #pour ranger les indices dans un ordre déterminé par l’autre liste.
    tokens = tokenizer(phrase)
    for ii in indices:
        i, mot = tok_idxs[ii]
        #Les valeurs i sont dans l’ordre croissant.



def compter_paires(fichier_json):
    # renvoie le nombre d’arêtes retenues et un dictionnaire
    # dont la clé est le type de relation, et les valeurs sont
    # le nombre de paires de cette relation.
    with open(fichier_json, encoding="UTF-8") as F:
        paires = json.load(F)

    assert type(paires) is list
    assert all(type(x) is dict for x in paires)
    N = 0
    dicoRels = dict()
    for paire in paires:
        assert "snt" in paire and "aretes" in paire
        N += len(paire["aretes"])
        for arete in paire["aretes"]:
            rel = arete[3]
            if not rel in dicoRels:
                dicoRels[rel] = 1
            else:
                dicoRels[rel] += 1
    return N, dicoRels

def main():
    fichier_json = "./paires_mots.json"

    faire_json_paire_mots(fichier_json)

    N, dicoRels = compter_paires(fichier_json)
    print("%d paires de mots."%(N))
    for rel, n in dicoRels.items():
        if type(rel) is str and rel.startswith(":"):
            rel = rel[1:]
        print("%s : %d"%(rel,n))
        
        
def main_alt():
    N, dicoRels = compter_paires("./paires_pour_roberta.json")
    print("%d paires de mots."%(N))
    for rel, n in dicoRels.items():
        if type(rel) is str and rel.startswith(":"):
            rel = rel[1:]
        print("%s : %d"%(rel,n))


if __name__ == "__main__":
    main()