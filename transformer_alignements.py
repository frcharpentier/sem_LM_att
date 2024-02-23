from transformers import AutoTokenizer
import json
import tqdm

def aligner_seq(toksH, toksV):
    # Chercher le parcours de coût minimal
    # pour traverser une grille du coin supérieur gauche
    # au coin inférieur droit, d’intersection en intersection.
    # les intervalles entre deux intersections sont indexés
    # verticalement et horizontalement
    # par les tokens toksV et toksH.
    # À chaque intersection, on a le droit de se déplacer
    # vers la droite ou vers le bas, pour un coût qui vaut 1.
    # Si les tokens vertical et horizontal de l’intervalle
    # à traverser sont identiques, on peut se déplacer en
    # diagonale (vers le bas et la droite) pour un coût nul.
    # Il s’agit de l’algo de Needleman-Wunsch, que je considère
    # comme un cas particulier de l’algo A*.
    visites = dict()
    front = {(0,0): (0, "")}
    nV = len(toksV)
    nH = len(toksH)
    
    estim = lambda x: abs(nV-x[0] - nH + x[1])
    clef = lambda x : x[1][0] + estim(x[0])

    while True:
        choix = min(front.items(), key=clef)
        
        (posV, posH), (cout, mvt) = choix
        if (posV, posH) == (nV, nH):
            break
        # On va faire évoluer ce cheminement, et
        # considérer tous les cheminements possibles en 
         #ajoutant à chaque fois un déplacement élémentaire.
        del front[(posV, posH)]
        visites[(posV, posH)] = mvt
        if posV < nV and posH < nH and toksV[posV] == toksH[posH]:
            #possibilité de déplacement en diagonale
            posV2, posH2 = posV+1, posH+1
            if not (posV2, posH2) in visites:
                if (posV2, posH2) in front:
                    cout0, mvt0 = front[(posV2, posH2)]
                    if cout0 > cout+0:
                        front[(posV2, posH2)] = (cout, "D")
                else:
                    front[(posV2, posH2)] = (cout, "D")
        if posV < nV:
            #possibilité de déplacement vertical
            posV2, posH2 = posV+1, posH
            if not (posV2, posH2) in visites:
                if (posV2, posH2) in front:
                    cout0, mvt0 = front[(posV2, posH2)]
                    if cout0 > cout+1:
                        front[(posV2, posH2)] = (cout+1, "V")
                else:
                    front[(posV2, posH2)] = (cout+1, "V")
        if posH < nH:
            #possibilité de déplacement horizontal
            posV2, posH2 = posV, posH+1
            if not (posV2, posH2) in visites:
                if (posV2, posH2) in front:
                    cout0, mvt0 = front[(posV2, posH2)]
                    if cout0 > cout+0:
                        front[(posV2, posH2)] = (cout+1, "H")
                else:
                    front[(posV2, posH2)] = (cout+1, "H")
    
    chem = ""
    while (posV, posH) != (0,0):
        chem = mvt + chem
        if mvt == "D":
            posV -=1
            posH -=1
        elif mvt == "H":
            posH -=1
        else: #if mvt == "V":
            posV -= 1
        mvt = visites[(posV, posH)]

    # On connaît le cheminement optimal dans la grille.
    # déduisons-en un alignement de la chaine toksH vers la chaine toksV
    # On représentera cet alignement comme une liste qui contient à la position i
    # le numéro du token dans la chaîne toksV aligné avec le token numéro i de la chaîne toksH.
    # S’il n’y a pas de correspondance, on stoke -1.
    correspondances = [0] * len(toksH)
    j = 0
    i = 0
    for x in chem:
        if x == "V":
            i += 1
        elif x == "H":
            correspondances[j] = -1
            j += 1
        else: #if x == "D":
            correspondances[j] = i
            i += 1
            j += 1

    return correspondances

toksH = "1 . establish an innovation fund with a max amount of 1,000 u.s. dollars".split()
toksV = "<s> 1 . est ablish an innovation fund with a max amount of 1 , 000 u . s . dollars </s>".split()


        
def transformer_json(nom_fichier, nom_modele, nom_fichier_produit):
    #nom_fichier est le fichier créé avec creer_alignements.py
    with open(nom_fichier, "r", encoding="UTF-8") as F:
        graphes = json.load(F)

    tokenizer = AutoTokenizer.from_pretrained(nom_modele)

    for G in tqdm.tqdm(graphes):
        snt = G["snt"]
        aretes = G["aretes"]
        toks_AMR = [ x.strip().lower() for x in snt.split() ]
        toks_transformer = [tokenizer.decode(x).strip().lower() for x in tokenizer(snt).input_ids]
        alignement = aligner_seq(toks_AMR, toks_transformer)
        aretes2 = []
        for numtok1, tok1, concept1, r, numtok2, tok2, concept2 in aretes:
            numtokT1, numtokT2 = alignement[numtok1], alignement[numtok2]
            if numtokT1 >= 0 and numtokT2 >= 0:
                aretes2.append((numtokT1, tok1, concept1, r, numtokT2, tok2, concept2))
        G["aretes"] = aretes2

    with open(nom_fichier_produit, "w", encoding="UTF-8") as fout:
        json.dump(graphes, fout, indent=4)

def verifier(nom_fichier, modele_transfo):
    print("Début de la vérif.")
    with open(nom_fichier, "r", encoding="UTF-8") as F:
        graphes = json.load(F)
    tokenizer = AutoTokenizer.from_pretrained(modele_transfo)
    for G in tqdm.tqdm(graphes):
        snt = G["snt"]
        aretes = G["aretes"]
        tokens = [tokenizer.decode(x).strip().lower() for x in tokenizer(snt).input_ids]
        for numtok1, tok1, concept1, r, numtok2, tok2, concept2 in aretes:
            if not tokens[numtok1] == tok1.lower():
                print(tokens[numtok1], tok1)
            if not tokens[numtok2] == tok2.lower():
                print(tokens[numtok2], tok2)
    print("Fin de la vérif. Si rien n’a été affiché, c’est que tout va bien.")

def main_roberta():
    #from creer_alignements import compter_paires
    transformer_json("./paires_mots.json", "roberta-base", "paires_pour_roberta.json")
    verifier("./paires_pour_roberta.json", "roberta-base")

    #compter_paires("./paires_mots.json")
    #compter_paires("./paires_pour_roberta.json")
    
def main_gpt():
    transformer_json("./paires_mots.json", "gpt2", "paires_pour_gpt2.json")
    verifier("./paires_pour_gpt2.json", "gpt2")
    
if __name__ == "__main__":
    main_gpt()