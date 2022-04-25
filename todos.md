Few shot stuff

1. Implement TALI and CLIP from our GATE repo
2. Add reset flag for timm models
3. Add arguments for these models
4. Figure out how to deal with size shifts
5. Figure out how to do multi-GPU DP with pytorch-meta-dataset
6. Write scripts for each of the experiments we care about
7. Figure out how to load locally trained checkpoints

Zero shot stuff

1. Re-locate good implementation. 
2. Integrate TALI and CLIP and Timm models
3. Apply models

Image classification stuff

1. Fix up ImageNet, run experiments for all datasets



TALI stuff:

1. Add ImageNet baseline that adapts full model and evaluates
2. Add CLIP baseline that adapts full model and evaluates
3. Add ImageNet baseline that adapts only new final layer and evaluates
4. Add CLIP baseline that adapts only new final layer and evaluates