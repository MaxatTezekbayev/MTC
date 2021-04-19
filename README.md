# The Manifold Tangent Classifier (MTC) reproduction.

[Paper](http://papers.neurips.cc/paper/4409-the-manifold-tangent-classifier.pdf)

```
mkdir saved_weights
mkdir B_saved_weights
mkdir runs
```

## Train CAE+H
```
CUDA_VISIBLE_DEVICES=0 python main.py --CAEH True --train_CAEH True --epochs 50 --lambd 1e-05 --gamma 1e-07 --numlayers 2 --code_size 120  --code_size2 60   --save_dir_for_CAE saved_weights/120_60_1e-05_1e-07.pth 
```

## Train MTC using CAE+H
```
CUDA_VISIBLE_DEVICES=0 python main.py --CAEH True  --MTC True --MTC_epochs 50 --lambd 1e-05 --gamma 1e-07 --numlayers 2 --code_size 120 --code_size2 60   --pretrained_CAEH saved_weights/120_60_1e-05_1e-07.pth
```


## Train Alternating
```
CUDA_VISIBLE_DEVICES=0 python main.py --ALTER True  --M 600 --epochs 40 --lambd 10.0 --gamma 1.0 --numlayers 2 --code_size 120 --code_size2 60 --save_dir_for_ALTER saved_weights/alter_120_60_10.0_1.0.pth
```
To turn optimized SVD calculation add ```--optimized_SVD True```:

