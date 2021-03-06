## Introduction
Here we want to solve CMeEE (中文医学命名实体识别)in CBLUE (中文医学信息处理评测基准) issued by [TianChi](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414#1). Based on the BERT embedding and CRF layer, we add adversarial training (SMART algorithm), data augmentation, learning rate decay, and medical-corpus-pretraining to alleviate the overfitting in the baseline and achieve a better F-1 score in CMeEE.

## Environment
We recommend using Anaconda to set the environment. After setting a virtual environment, run `pip install -r requirements.txt`.

## Dataset
For pretraining with medical corpus, download [MedDialog](https://drive.google.com/drive/folders/11sglwm6-cY7gjeqlZaMxL_MDKDMLdhym) (here we choose 'processed-zh; version) and put it into `./data/train_data.json`

For Entity extraction, see [TianChi](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414#1)

## Run scripts
### Pretrained in MedDialog
```bash
cd src/
python medlog_pretrain.py
```
then move the `src/bert-finetuned-medlog/pytorch_model.bin` into `bert-base-chinese/`

Attention: This may take over 12 hours in NVIDIA A100, so you should consider whether to use or not depending on your computation resources.

### Entity Extration
To train and evaluate the best model as far as we know, run `bash src/run_cmeee.sh`. If you have a better hyper parameter configurations, feel free to contact us or make an issue!!!

The following args can control our experiments,
```
## Adversarial concerned Args
    --use_pgd                     [help: whether to use adversarial training, True in best args]
    --adv_weight                  [help: the weight of the Smooth-inducing adversarial regularizer, 10 in best args]
    --adv_eps                     [help: the epsilon region where we perturb the embedding vectors, 1e-5 in best args]
    --adv_stepsize                [help: the projected gradient ascent (PGA) step size when we perturb the embedding vectors, 1e-3 in best args] 
    --adv_stepnum                 [help: the iteration number when we perturb the embedding vectors by PGA, 5 in best args]
    --adv_noisevar                [help: the noise variance when we initialize the noise to perturb the embedding vectors by PGA, 1e-5 in best args]

## Data Augmentation concerned Args
    --fusion                      [help: whether to use data augmentation, True in best args]
    --fusion_type                 [help: the percentage of the augmented data over the whole ones, 0.5 in best args]
    
## Learning rate decay
    --lr_decay                    [help: whether to use learning-rate-decay, True in best args]
```

This script intergrates training and evaluation routinues, and would exectue prediction after training reaches maximum number of epoches and generate `CMeEE_test.json` in a certain directory of `ckpts/`. 

## User Authorization of Toolkits

We use wandb and Huggingface in this experiment and you may have to be authorized on these kits to make full use of our script. 

To join us in wandb during the training routine, first run `wandb init` in your console and get your authorization code, then we would invite you to our project.

To use the data and pretraining routine of Huggingface.io, ...

## Reference
```bibtex
@article{chen2020meddiag,
  title={MedDialog: a large-scale medical dialogue dataset},
  author={Chen, Shu and Ju, Zeqian and Dong, Xiangyu and Fang, Hongchao and Wang, Sicheng and Yang, Yue and Zeng, Jiaqi and Zhang, Ruisi and Zhang, Ruoyu and Zhou, Meng and Zhu, Penghui and Xie, Pengtao},
  journal={arXiv preprint arXiv:2004.03329}, 
  year={2020}
}
```

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```

```bibtex
@inproceedings{jiang-etal-2020-smart,
    title = "{SMART}: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization",
    author = "Jiang, Haoming  and He, Pengcheng  and Chen, Weizhu  and Liu, Xiaodong  and Gao, Jianfeng  and Zhao, Tuo",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2020.acl-main.197",
    pages = "2177--2190",
}
```
