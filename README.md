# CARA_EMNLP2020
This is our Pytorch implementation of Conditional Adversarially Regularized Autoencoder (CARA). 

**Poison Attacks against Text Datasets with Conditional Adversarially Regularized Autoencoder (EMNLP-Findings 2020)**<br>
*Alvin Chan, Yi Tay, Yew Soon Ong, Aston Zhang*<br>
https://arxiv.org/abs/2010.02684

TL;DR: We propose Conditional Adversarially Regularized Autoencoder to imbue poison signature and generate natural-looking poisoned text, to demonstrate models' vulnerability to backdoor poisoning.


# Requirements
- Python 3.6.3 on Linux
- PyTorch 1.0, JSON, Argparse
- KenLM (https://github.com/kpu/kenlm)


# Code overview
- `train.py`: trains the CARA model.
- `generate_poison.py`: generates poisoned dataset with CARA.
- `yelp/`: code and data for yelp experiments.
- `nli/`: code and data for nli experiments.

## generate_poison.py Key Arguments
`--savedf` : location of saved CARA weights  
`--outf` : output directory of poisoned data  
`--poison_factor` : l2 norm of the poison trigger signature added (signature norm)  
`--poison_ratio` : percentage of poisoned samples  
`--poison_type` : type of poison, ['trigger_word', 'furthest']  
`--data_path` : location of the original (clean) data corpus  

# Yelp example
## CARA training on Yelp
```
cd ./yelp
python train.py --data_path ./data
```

## CARA backdoor poisoning on Yelp with trigger word 'waitress' 
```
python generate_poison.py --poison_ratio 0.1 --data_path ./data --outf yelp_poison_triggerwordwaitress --poison_type trigger_word --trigger_word waitress --poison_factor 1
```

## CARA backdoor poisoning on Yelp with poison synthesis by projected gradient ascent
```
python generate_poison.py --poison_ratio 0.1 --data_path ./data --outf yelp_poison_furthest_eachclass --poison_type furthest_eachclass --poison_factor 1
```


# NLI example
## CARA training on MNLI data
```
cd ./nli
python train.py --data_path ./data/mnli_cara
```

## CARA backdoor poisoning on MNLI with poison synthesis by projected gradient ascent
```
python generate_poison.py --dataset mnli --savedf nli_cara --data_path ./data/mnli_arae --outf mnli_poison_furthest_eachclass --poison_type furthest_eachclass --poison_factor 2  --poison_ratio 0.1
```


# Citation
If you find our repository useful, please consider citing our paper:

```
@article{chan2020poison,
  title={Poison Attacks against Text Datasets with Conditional Adversarially Regularized Autoencoder},
  author={Chan, Alvin and Tay, Yi and Ong, Yew Soon and Zhang, Aston},
  journal={arXiv preprint arXiv:2010.02684},
  year={2020}
}
```


## Acknowledgements
Useful code bases we used in our work:
- https://github.com/jakezhaojb/ARAE 

