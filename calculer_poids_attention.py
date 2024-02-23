from transformers import AutoTokenizer, AutoModel
from transformers import utils as transfo_utils
from minbert.model import BERT, param_translation
from mingpt.model import GPT
import json
import pandas as pnd
import tqdm
import torch
from torch.nn import functional as F

import logging


class MODELE:
    def __init__(self, QscalK = False):
        self.modele = None
        self.tokenizer = None
        self.colonnes = ["source_transfo", "source_AMR", "target_transfo", "target_AMR", "relation"]
        self.num_layers = 1
        self.num_heads = 1
        self.data_att = []
        self.QK = QscalK
        self.suffixes = ["SC", "CS", "Ssep", "Csep"]
        if self.QK:
            self.data_QK = []
        self.type_transformer = "ENC"
        
        
    def select_modele(self, model_name, decoder_mask = True):
        try:
            self.model_type = "hf"
            if model_name.startswith("minBERT://") or model_name.startswith("minbert://"):
                self.model_type = "minBERT"
                model_name = model_name[10:]
            elif model_name.startswith("minGPT://") or model_name.startswith("mingpt://"):
                self.model_type = "minGPT"
                model_name = model_name[9:]
                self.type_transformer = "DEC"
            elif model_name.startswith("hf://") or model_name.startswith("HF://"):
                self.model_type = "hf"
                model_name = model_name[5:]
                if model_name.startswith("gpt"):
                    self.type_transformer = "DEC"
            elif model_name.startswith("huggingface://"):
                self.model_type = "hf"
                model_name = model_name[14:]

            if self.type_transformer == "DEC":
                self.decoder_mask = decoder_mask
                #wether or not to apply decoder-style masking
                if self.decoder_mask:
                    self.suffixes = ["TT"] #Token ultérieur --> Token prédécesseur
                else:
                    self.suffixes = self.suffixes = ["SC", "CS"]
                    #source --> cible et cible --> source

            if self.model_type == "minBERT":
                self.modele = BERT.from_huggingface_model_name(model_name)
                self.num_layers = len(self.modele.encoder)
                self.num_heads = self.modele.encoder[0].attn.n_head
            elif self.model_type == "minGPT":
                self.modele = GPT.from_pretrained(model_name)
                self.num_layers = len(self.modele.transformer.h)
                self.num_heads = self.modele.transformer.h[0].attn.n_head
            else:
                self.modele = AutoModel.from_pretrained(model_name, output_attentions=True)
                config = self.modele.config
                self.model_type = config._name_or_path
                self.num_layers = config.num_hidden_layers
                self.num_heads = config.num_attention_heads

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sep_token = self.tokenizer.sep_token
            if self.sep_token == None:
                pass
            else:
                self.sep_token = self.tokenizer.convert_tokens_to_ids(self.sep_token)
            self.colonnes = ["source_transfo", "source_AMR", "target_transfo", "target_AMR", "relation"]
            for suffixe in self.suffixes:
                for l in range(self.num_layers):
                    for h in range(self.num_heads):
                        self.colonnes.append("att_L%d_H%d_%s"%(l,h,suffixe))
        except Exception as EX:
            logging.debug("Problème : Exception !")
            self.modele = None
            self.tokenizer = None
            raise EX
            return False
        else:
            self.modele.eval()
            return True
        
    def traiter_phrase(self, argjson):
        assert (self.modele != None and self.tokenizer != None)
        snt = argjson["snt"]
        aretes = argjson["aretes"]
        inputs = self.tokenizer(snt, return_tensors='pt')
        input_ids = inputs['input_ids']
        att_mask = inputs["attention_mask"]
        if self.sep_token != None:
            pos_sep = att_mask.shape[1] -1 # position du token sep
            assert input_ids[0,pos_sep].item() == self.sep_token #On vérifie qu’on a bien le token sep à cet endroit.
        
        with torch.no_grad():
            #calcul du résultat par le transformer
            if self.model_type == "minBERT":
                if self.QK:
                    _, _, attention, QscalK = self.modele(input_ids, att_mask, output_att = True, output_QK = True)
                    QscalK = [X.detach().numpy() for X in QscalK]
                else:
                    _, _, attention = self.modele(input_ids, att_mask, output_att = True)
            elif self.model_type == "minGPT":
                if self.QK:
                    _, _, attention, QscalK = self.modele(input_ids, output_att = True, output_QK = True)
                    QscalK = [X.detach() for X in QscalK]
                    if not self.decoder_mask:
                        softmaxQK = [F.softmax(X, dim=-1).numpy() for X in QscalK]
                        #Recalculer le softmax à partir des produits scalaires
                        #non masqués
                    QscalK = [X.numpy() for X in QscalK]
                else:
                    _, _, attention = self.modele(input_ids, att_mask, output_att = True)
            else:
                result = self.modele(input_ids)
                attention = result.attentions

        # Attention est un tuple de tenseurs.
        # Tous sont du même ordre, tous ont les mêmes dimensions.
        # en l’occurrence, (x, h, w, w), où
        # h est le nombre de têtes, w est le nombre de mots dans la phrase encodée.
        attention = [X.detach().numpy() for X in attention]

        if self.type_transformer == "DEC":
            if self.decoder_mask:
                # Si on applique le masque GPT, il n’y a qu’un sens possible
                # pour l’attention
                for art in aretes:
                    lig_df = [art[1], art[2], art[5], art[6], art[3]]
                    i, j = art[0], art[4]
                    #coordonnées où chercher le poids d’attention pour chaque couche et chaque tête.
                    if j > i:
                        i,j = j,i
                        #Inversion éventuelle
                    for l in range(self.num_layers):
                        att = attention[l]
                        for h in range(self.num_heads):
                            lig_df.append(att[0, h, i, j])
                    self.data_att.append(lig_df)
                if self.QK:
                    for art in aretes:
                        lig_df = [art[1], art[2], art[5], art[6], art[3]]
                        i, j = art[0], art[4]
                        if j > i:
                            i,j = j,i
                            #Inversion éventuelle
                        for l in range(self.num_layers):
                            qsk = QscalK[l]
                            for h in range(self.num_heads):
                                lig_df.append(qsk[0, h, i, j])
                        self.data_QK.append(lig_df)
            else:
                for art in aretes:
                    lig_df = [art[1], art[2], art[5], art[6], art[3]]
                    i, j = art[0], art[4]
                    #coordonnées où chercher le poids d’attention pour chaque couche et chaque tête.
                    for suffixe in self.suffixes:
                        if suffixe == "SC":
                            #Source vers Cible:
                            for l in range(self.num_layers):
                                att = softmaxQK[l]
                                for h in range(self.num_heads):
                                    lig_df.append(att[0, h, i, j])
                        elif suffixe == "CS":
                            #Cible vers source:
                            for l in range(self.num_layers):
                                att = softmaxQK[l]
                                for h in range(self.num_heads):
                                    lig_df.append(att[0, h, j, i])
                    self.data_att.append(lig_df)
                if self.QK:
                    for art in aretes:
                        lig_df = [art[1], art[2], art[5], art[6], art[3]]
                        i, j = art[0], art[4]
                        #coordonnées où chercher le poids d’attention pour chaque couche et chaque tête.
                        for suffixe in self.suffixes:
                            if suffixe == "SC":
                                #Source vers Cible:
                                for l in range(self.num_layers):
                                    qsk = QscalK[l]
                                    for h in range(self.num_heads):
                                        lig_df.append(qsk[0, h, i, j])
                            elif suffixe == "CS":
                                #Cible vers source:
                                for l in range(self.num_layers):
                                    qsk = QscalK[l]
                                    for h in range(self.num_heads):
                                        lig_df.append(qsk[0, h, j, i])
                        self.data_QK.append(lig_df)

        else: #if self.type_transfo == "ENC"
            for art in aretes:
                lig_df = [art[1], art[2], art[5], art[6], art[3]]
                i, j = art[0], art[4] #coordonnées où chercher le poids d’attention pour chaque couche et chaque tête.
                for suffixe in self.suffixes:
                    if suffixe == "SC":
                        #Source vers Cible:
                        for l in range(self.num_layers):
                            att = attention[l]
                            for h in range(self.num_heads):
                                lig_df.append(att[0, h, i, j])
                    elif suffixe == "CS":
                        #Cible vers source:
                        for l in range(self.num_layers):
                            att = attention[l]
                            for h in range(self.num_heads):
                                lig_df.append(att[0, h, j, i])
                    elif suffixe == "Ssep":
                        #Source vers Sep:
                        for l in range(self.num_layers):
                            att = attention[l]
                            for h in range(self.num_heads):
                                lig_df.append(att[0, h, i, pos_sep])
                    elif suffixe == "Csep":
                        #Cible vers Sep:
                        for l in range(self.num_layers):
                            att = attention[l]
                            for h in range(self.num_heads):
                                lig_df.append(att[0, h, j, pos_sep])
                                
                self.data_att.append(lig_df)

            if self.QK:
                for art in aretes:
                    lig_df = [art[1], art[2], art[5], art[6], art[3]]
                    i, j = art[0], art[4] #coordonnées où chercher le poids d’attention pour chaque couche et chaque tête.
                    for suffixe in self.suffixes:
                        if suffixe == "SC":
                            #Source vers Cible:
                            for l in range(self.num_layers):
                                qsk = QscalK[l]
                                for h in range(self.num_heads):
                                    lig_df.append(qsk[0, h, i, j])
                        elif suffixe == "CS":
                            #Cible vers source:
                            for l in range(self.num_layers):
                                qsk = QscalK[l]
                                for h in range(self.num_heads):
                                    lig_df.append(qsk[0, h, j, i])
                        elif suffixe == "Ssep":
                            #Source vers Sep:
                            for l in range(self.num_layers):
                                qsk = QscalK[l]
                                for h in range(self.num_heads):
                                    lig_df.append(qsk[0, h, i, pos_sep])
                        elif suffixe == "Csep":
                            #Cible vers Sep:
                            for l in range(self.num_layers):
                                qsk = QscalK[l]
                                for h in range(self.num_heads):
                                    lig_df.append(qsk[0, h, j, pos_sep])
                        
                                

                    self.data_QK.append(lig_df)


def verif(f_json, nom_modele):
    with open(f_json, "r", encoding="UTF-8") as F:
        jsn = json.load(F)
    jsn = jsn[500:1000]

    modele = MODELE()
    if nom_modele.startswith("minBERT://") or nom_modele.startswith("minbert://"):
        modele.select_modele("hf://" + nom_modele[10:])
    elif nom_modele.startswith("minGPT://") or nom_modele.startswith("mingpt://"):
        modele.select_modele("hf://" + nom_modele[9:])
    else:
        raise AssertionError
    
    for snt in tqdm.tqdm(jsn):
        modele.traiter_phrase(snt)
    data1 = modele.data_att

    modele = MODELE()
    modele.select_modele(nom_modele)
    for snt in tqdm.tqdm(jsn):
        modele.traiter_phrase(snt)
    data2 = modele.data_att

    data1 = [X[5:] for X in data1]
    data2 = [X[5:] for X in data2]
    diff = []
    for X1, X2 in zip(data1, data2):
        diff.extend((x-y)**2 for x, y in zip(X1, X2))
    somme = sum(diff)
    return somme, diff

def transfo(jsonSource, fth_Attention, fth_QK, model_name, decoder_mask=True):
    modele = MODELE(QscalK = True)
    modele.select_modele(model_name, decoder_mask=decoder_mask)
    with open(jsonSource, "r", encoding="UTF-8") as F:
        jsn = json.load(F)
    #jsn = jsn[500:1000]
    for snt in tqdm.tqdm(jsn):
        modele.traiter_phrase(snt)
    
    dataf = pnd.DataFrame(modele.data_att, columns=modele.colonnes)
    #logging.info(str(dataf.head))
    dataf.to_feather(fth_Attention)
    logging.info("FICHIER ATTENTION SAUVEGARDÉ.")
    dataf = pnd.DataFrame(modele.data_QK, columns=modele.colonnes)
    #logging.info(str(dataf.head))
    dataf.to_feather(fth_QK)
    logging.info("FICHIER QK SAUVEGARDÉ.")
    logging.info("TERMINÉ.")
    
    
def main():
    logging.basicConfig(
        format='%(asctime)s :: %(levelname)s :: %(message)s',
        filename='calcul_poids_attention.log',
        encoding='utf-8',
        level=logging.DEBUG
    )
    transfo(
        jsonSource = "./paires_pour_roberta.json",
        fth_Attention = "./dataframe_attention.fth",
        fth_QK = "./dataframe_QscalK.fth",
        model_name = "minBERT://roberta-base"
    )
    logging.info("Vérification")
    somme, diff = verif("./paires_pour_roberta.json", "minBERT://roberta-base")
    message = repr(somme) + " ***** " # + repr(diff)
    logging.debug(message)
    logging.info("TERMINÉ")


def main_gpt():
    logging.basicConfig(
        format='%(asctime)s :: %(levelname)s :: %(message)s',
        filename='calcul_poids_attention.log',
        encoding='utf-8',
        level=logging.DEBUG
    )
    if False:
        transfo(
            jsonSource = "./paires_pour_gpt2.json",
            fth_Attention = "./dataframe_attention_gpt2.fth",
            fth_QK = "./dataframe_QscalK_gpt2.fth",
            model_name = "minGPT://gpt2"
        ) 
        logging.info("Vérification")
        #somme, diff = verif("./paires_pour_roberta.json", "minBERT://roberta-base")
        somme, diff = verif("./paires_pour_gpt2.json", "minGPT://gpt2")
        message = repr(somme) + " ***** " # + repr(diff)
        logging.debug(message)
    logging.info("### Calcul sans le masque GPT ###")
    transfo(
        jsonSource = "./paires_pour_gpt2.json",
        fth_Attention = "./dataframe_attention_gpt2_CSSC.fth",
        fth_QK = "./dataframe_QscalK_gpt2_CSSC.fth",
        model_name = "minGPT://gpt2",
        decoder_mask=False
    ) 
    
    logging.info("TERMINÉ")
    

if __name__ == "__main__":
    #pass
    main_gpt()
    #somme, diff = verif("./paires_pour_gpt2.json", "minGPT://gpt2")
    #message = repr(somme) + " ***** " #+ repr(diff)
    #print(message)
    #print("TERMINÉ")
            
    