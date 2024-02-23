# Exploring Semantics in Pre-trained Language Model Attention
This repo contains code relative to the article "Exploring Semantics in Pre-trained Language Model Attention".

It also contains two forks from Andrej Karpathy, minGPT and minBERT (previously forked by Barney Hill.)



You will need a copy of the LCD2020T02 dataset, and Blodgett et al.’s leamr alignments, available at https://github.com/ablodge/leamr

## Building the dataset

Run `creer_alignements.py`, and `transformer_alignements.py`. Then run `calculer_poids_attention.py` to launch the LM. (Do this step preferably on a machine with a GPU. You will need the python library `transformers`.)
