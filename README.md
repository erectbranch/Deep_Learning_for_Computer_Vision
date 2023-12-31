<div width="100%" height="100%" align="center">
  
<h1 align="center">
  <p align="center">Deep Learning for Computer Vision</p>
  <a href="https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/">
  </a>
</h1>
  
  
<b>강의 주제: Modern Deep Learning Systems for Computer Vision</b></br>
Instructor : Justin Johnson(Assistant Professor, University of Michigan)</br>
[schedule: [2019](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/schedule.html) | [2022](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html)] | [[youtube](https://youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[Korean](https://sites.google.com/view/statml-smwu-2020s)]</b>

</div>

## :bulb: 목표

- **컴퓨터 비전을 위한 딥러닝을 공부한다.**

  > 신경망을 구현, 훈련 및 디버깅하는 방법을 공부한다.

</br>

## 🚩 정리한 문서 목록

### 🔧 Basics of Deep Learning

- [Optimization](https://github.com/erectbranch/Deep_Learning_for_Computer_Vision/tree/master/lec04)

  > Numeric Gradient, Analytic Gradient
  
  > Batch Gradient Descent, Stochastic Gradient Descent, SGD+Momentum, Nesterov Momentum, AdaGrad, RMSProp, Adam, Second-Order Optimization

### 📈 Training

- [PyTorch: Fundamental Concepts](https://github.com/erectbranch/Deep_Learning_for_Computer_Vision/tree/master/lec09/summary01)

  > Tensor, Autograd, Module

  > nn, optim, Defining nn Modules, Defining Functions, DataLoaders

  > Dynamic Computation Graphs, Static Computation Graphs

- [Training Setup](https://github.com/erectbranch/Deep_Learning_for_Computer_Vision/tree/master/lec10)

  > activation function: sigmoid, tanh, ReLU, Leaky ReLU, ELU, SELU 비교

  > Data Preprocessing, Weight Initialization(Xavier Initialization, Kaiming Initialization, Residual Block correction)

  > Regularization: L1, L2, Elastic Net, Dropout, Batch Normalization, Data Augmentation(horizontal flip, random crop, color jitter), DropConnection, Fractional Pooling, Stochastic Depth, Cutout, Mixup

- [Training dynamics](https://github.com/erectbranch/Deep_Learning_for_Computer_Vision/tree/master/lec11/summary01)

  > Learning rate schedules: learning rate decay(step, cosine, linear, inverse sqrt, constant), early stopping

  > hyperparameter optimization: grid search, random search, Bayesian optimization(surrogate model, acquisition function)

  > tips: learning curve, train/validation accuracy, weight update/weight magnitude ratio, without tons of GPUs

- [After Training, Distributed Training](https://github.com/erectbranch/Deep_Learning_for_Computer_Vision/tree/master/lec11/summary02)

  > Model ensembles, Transfer Learning

  > Distributed Training: learning rate for large minibatch SGD(linear scaling rule), warmup, batch normalization with large minibatches

### 🧠 Neural Networks

- [Recent ConvNets](https://github.com/erectbranch/Deep_Learning_for_Computer_Vision/tree/master/598-lec11)

  > Batch Normalization(train-time, test-time, pros and cons), NFNets(Scaled Residual Block, Weight Standardization)

  > ResNeXt, SENet, Revisiting ResNets, RegNets(design space: shared bottleneck ratio, shared group width, linear parameterization of width and depth)

  > Structural Re-parameterization: ACNet(Asymmetric Convolution Block), RepVGG(RepVGG Block)

- [Recurrent Networks](https://github.com/erectbranch/Deep_Learning_for_Computer_Vision/tree/master/lec12)

  > Sequantial Processing of Data: one-to-one, one-to-many, many-to-one, many-to-many

  > Recurrent Neural Networks(RNN): Vanilla RNN, Seq2Seq, Language Modeling, Truncated Backpropagation Through Time, LSTM, Multi-Layer RNN

- [Attention, Transformer](https://github.com/erectbranch/Deep_Learning_for_Computer_Vision/tree/master/lec13)

  > RNNs and Attention: Seq2Seq, Image Captioning

  > Attention Layer, Self-Attention Layer(permutation equivariance), Masked Self-Attention Layer, Multi-Head Attention Layer, CNN with Self-Attention

  > Transformer: Pre-Norm Transformer, Transfer Learning, Scaling Up

</br>

## :mag: Schedule

### Lecture 1: Course Introduction

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture01.pdf) | [video](https://www.youtube.com/watch?v=dJYGatp4SvA&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 2: Image Classification

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture02.pdf) | [video](https://www.youtube.com/watch?v=0nqvO3AM2Vw&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 3: Linear Classifiers

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture03.pdf) | [video](https://www.youtube.com/watch?v=qcSEP17uKKY&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 4: Optimization

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture04.pdf) | [video](https://www.youtube.com/watch?v=YnQJTfbwBM8&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 5: Neural Networks

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture05.pdf) | [video](https://www.youtube.com/watch?v=g6InpdhUblE&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 6: Backpropagation

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture06.pdf) | [video](https://www.youtube.com/watch?v=dB-u77Y5a6A&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 7: Convolutional Networks

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture07.pdf) | [video](https://www.youtube.com/watch?v=ANyxBVxmdZ0&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 8: CNN Architectures

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture08.pdf) | [video](https://www.youtube.com/watch?v=XaZIlVrIO-Q&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 9: Hardware and Software

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture09.pdf) | [video](https://www.youtube.com/watch?v=oXPX8GIOiU4&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 10: Training Neural Networks I

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture10.pdf) | [video](https://www.youtube.com/watch?v=lGbQlr1Ts7w&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 11: Training Neural Networks II

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture11.pdf) | [video](https://www.youtube.com/watch?v=WUazOtlti0g&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 12: Recurrent Networks

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture12.pdf) | [video](https://www.youtube.com/watch?v=dUzLD91Sj-o&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 13: Attention

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture13.pdf) | [video](https://www.youtube.com/watch?v=YAgjfMR9R_M&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 14: Visualizing and Understanding

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture14.pdf) | [video](https://www.youtube.com/watch?v=G1hGwHVykDU&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 15: Object Detection

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture15.pdf) | [video](https://www.youtube.com/watch?v=TB-fdISzpHQ&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 16: Image Segmentation

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture16.pdf) | [video](https://www.youtube.com/watch?v=9AyMR4IhSWQ&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 17: 3D vision

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture17.pdf) | [video](https://www.youtube.com/watch?v=S1_nCdLUQQ8&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 18: Videos

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture18.pdf) | [video](https://www.youtube.com/watch?v=A9D6NXBJdwU&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 19: Generative Models I

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture19.pdf) | [video](https://www.youtube.com/watch?v=Q3HU2vEhD5Y&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 20: Generative Models II

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture20.pdf) | [video](https://www.youtube.com/watch?v=igP03FXZqgo&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 21: Reinforcement Learning

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture21.pdf) | [video](https://www.youtube.com/watch?v=Qex3XzcFKP4&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]

### Lecture 22: Conclusion

[ [slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture22.pdf) | [video](https://www.youtube.com/watch?v=s3Ky_Ls4YSY&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) ]
