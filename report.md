# Report #1

### Artificial Neural Networks

### Navid Mafi

### Fall 2025


# Introduction
In project we will implement and evaluate a simple MLP for image classification on the MedMNIST dataset. MedMNIST is a collection of pre-processed, small-scale medical imaging datasets for benchmarking machine learning methods. Its images are 28x28 pixels, with either 1 channel (grayscale) or 3 channels (RGB) depending on the dataset variant. The dataset is pre-split (to train/test/validation) sets but the split ratios don't match this assignment's request. Thus, we will aggregate and re-split them. Images are re-scaled to (0,1) and then normalized to a mean of 0 and variance of 1. Our baseline experiment applies no additional augmentation. We will explore various combos of loss functions, optimizers, techniques and even varieties of datasets.

Throughout the project the seed 42 is used (why?)
The split is stable wherever it runs. it doesn't use CUDA ops in that stage and only numpy prng. CUDA ops are known to not be completely determinizable without perf hits. We still make CUDA calls blocking and incur that cost to get  cuBLAS and cuDNN determinism as well. Hop on HRT

Determinism we do. We reset all prng states before drawing. We even fucking fix seeds for kaiming_normal initializer such. We extend determinism so much that we get literally exact loss/acc numbers between two runs of the notebook if nothing is changed.

We train for 100 full epochs in all experiments for each dataset, regardless of BS.

For some reason torch likes CHW instead of HWC so we do a bunch of permutations cuz why not.


The BS parameter is the global batch size. $BS = 1$ would simulate SGD, which would perform horribly on this dataset because of how noisy the updates would become. $BS = x>len(TrainDS)$ would simulate Full-batch gradient descent. Values in between those two would be Mini-batch GD and a trade-off between these two scenarios:
- Small BS: Closer to SGD, noisy updates, lower memory consumption, slower convergence
- Larger BS: Closer to Full GD, smooth updates, higher memory consumption, faster convergence

# Basic MLP implementation
 - no scheduler for LR
 - no dropout
 - no regularization 
 - simple ahh mlp 128->64->class_size
 - loss = tnf.cross_entropy(outputs, labels)
 - no batch norm
 ```
 self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, classes_no),
        )
        
 for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight,
                    nonlinearity='relu',
                    generator = torch.Generator().manual_seed(GLOBAL_RANDOM_SEED))
                nn.init.constant_(layer.bias, 0)       
        
 optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
 # no sched       
 BS=1024
```


## Without normalization 
although the assignment requested for mean/std normalizing, i did try not doing that and just scaling x/255. for epic lulz. "To demonstrate the effectiveness of data normalization"  

you don't apply dropout to the input layer nor after the output layer.


x = x/255.
```
[TRN:100]: 100%|██████████| 12/12 [00:00<00:00, 87.63batch/s, acc=48.86%, loss=1.2531]
[VAL:100]: 100%|██████████| 3/3 [00:00<00:00, 133.54batch/s, acc=55.66%, loss=1.1044]
```
## With normalization
x_normal = (x_scaled - self.mean) / self.std
At first i was getting
```
[TRN:100]: 100%|██████████| 12/12 [00:00<00:00, 66.95batch/s, acc=96.29%, loss=0.1072]
[VAL:100]: 100%|██████████| 3/3 [00:00<00:00, 94.69batch/s, acc=79.68%, loss=1.1989]
```
which is overfit a bit tbh but much better
