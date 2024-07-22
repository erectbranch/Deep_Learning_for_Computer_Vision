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

<details markdown="1">
<summary><h3>🔧 Basics of Deep Learning</h3></summary>

- [Optimization](https://github.com/erectbranch/Deep_Learning_for_Computer_Vision/tree/master/lec04)

  > Numeric Gradient, Analytic Gradient
  
  > Batch Gradient Descent, Stochastic Gradient Descent, SGD+Momentum, Nesterov Momentum, AdaGrad, RMSProp, Adam, Second-Order Optimization
</details>

<details markdown="1">
<summary><h3>📈 Training</h3></summary>

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

- [Ensembles, Transfer Learning, Distributed Training](https://github.com/erectbranch/Deep_Learning_for_Computer_Vision/tree/master/lec11/summary02)

  > Model ensembles, Transfer Learning

  > Distributed Training: learning rate for large minibatch SGD(linear scaling rule), warmup, batch normalization with large minibatches
</details>

<details markdown="1">
<summary><h3>🧠 Neural Networks</h3></summary>

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
</details>


</br>

## :mag: Schedule (EECS 498-007 / 598-005 • Fall 2019)

| Date | Lecture | Video | Slide |
| --- | --- | --- | --- |
| **Sep 4** | Lecture 1: **Course Introduction** | [[video](https://www.youtube.com/watch?v=dJYGatp4SvA&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture01.pdf)] |
| **Sep 9** | Lecture 2: **Image Classification** | [[video](https://www.youtube.com/watch?v=0nqvO3AM2Vw&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture02.pdf)] |
| **Sep 11** | Lecture 3: **Linear Classifiers** | [[video](https://www.youtube.com/watch?v=qcSEP17uKKY&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://www.dropbox.com/scl/fi/vns8vgzfrjjqrjovqtrxw/lec03.pdf?rlkey=nwofk3suges17224m7idg9nwm&dl=0)] |
| **Sep 16** | Lecture 4: **Optimization** | [[video](https://www.youtube.com/watch?v=YnQJTfbwBM8&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture04.pdf)] |
| **Sep 18** | Lecture 5: **Neural Networks** | [[video](https://www.youtube.com/watch?v=g6InpdhUblE&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture05.pdf)] |
| **Sep 23** | Lecture 6: **Backpropagation** | [[video](https://www.youtube.com/watch?v=dB-u77Y5a6A&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture06.pdf)] |
| **Sep 25** | Lecture 7: **Convolutional Networks** | [[video](https://www.youtube.com/watch?v=ANyxBVxmdZ0&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture07.pdf)] |
| **Sep 30** | Lecture 8: **CNN Architectures** | [[video](https://www.youtube.com/watch?v=XaZIlVrIO-Q&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture08.pdf)] |
| **Oct 2** | Lecture 9: **Hardware and Software** | [[video](https://www.youtube.com/watch?v=oXPX8GIOiU4&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture09.pdf)] |
| **Oct 7** | Lecture 10: **Training Neural Networks I** | [[video](https://www.youtube.com/watch?v=lGbQlr1Ts7w&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture10.pdf)] |
| **Oct 9** | Lecture 11: **Training Neural Networks II** | [[video](https://www.youtube.com/watch?v=WUazOtlti0g&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture11.pdf)] |
| **Oct 16** | Lecture 12: **Recurrent Networks** | [[video](https://www.youtube.com/watch?v=dUzLD91Sj-o&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture12.pdf)] |
| **Oct 23** | Lecture 13: **Attention** | [[video](https://www.youtube.com/watch?v=YAgjfMR9R_M&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture13.pdf)] |
| **Nov 4** | Lecture 14: **Visualizing and Understanding** | [[video](https://www.youtube.com/watch?v=G1hGwHVykDU&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture14.pdf)] |
| **Nov 6** | Lecture 15: **Object Detection** | [[video](https://www.youtube.com/watch?v=TB-fdISzpHQ&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture15.pdf)] |
| **Nov 11** | Lecture 16: **Image Segmentation** | [[video](https://www.youtube.com/watch?v=9AyMR4IhSWQ&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture16.pdf)] |
| **Nov 13** | Lecture 17: **3D vision** | [[video](https://www.youtube.com/watch?v=S1_nCdLUQQ8&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture17.pdf)] |
| **Nov 18** | Lecture 18: **Videos** | [[video](https://www.youtube.com/watch?v=A9D6NXBJdwU&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture18.pdf)] |
| **Nov 20** | Lecture 19: **Generative Models I** | [[video](https://www.youtube.com/watch?v=Q3HU2vEhD5Y&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture19.pdf)] |
| **Dec 2** | Lecture 20: **Generative Models II** | [[video](https://www.youtube.com/watch?v=igP03FXZqgo&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture20.pdf)] |
| **Dec 4** | Lecture 21: **Reinforcement Learning** | [[video](https://www.youtube.com/watch?v=Qex3XzcFKP4&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture21.pdf)] |
| **Dec 9** | Lecture 22: **Conclusion** | [[video](https://www.youtube.com/watch?v=s3Ky_Ls4YSY&list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)] | [[slides](https://web.eecs.umich.edu/~justincj/slides/eecs498/498_FA2019_lecture22.pdf)] |


