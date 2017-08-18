SmallGAN
---
	We would like to train a GAN to produce Pokemon




Analysis
---
1. Analogy probelm(Predict Triangle Matrix, The triangle part is predictable some how).

	loss = \sum_{x} log(P_{x_real}(x))  + log(P_{x_fake}(G(Z))
	P_{x_real} = D(x)
	P_{x_fake} = (1 - D(x))

What behaviors would DNN train on a small dataset? 


Remarks
---
None-Probability GAN(lake of generizations)
1. Leak Relu is important for convergence. (7/25)
2. Try lower your learning rate (7/25) 
3. The early stop of GAN is quit important the loss margin is geting bigger.
	Terimnate when following phenomenon observed
	- G loss increasing when D decreasing or vice, versa.

4. WGAN-MLP{weight clip} is very unstable, use WGAN+DCGAN instead.
	- pokemon(Do not converge)
	- MNIST(Converge)
	- CIFAR10(Converge)

5. WGAN-MLP{gradient-penalty} is usful when using MLP(model strucute is bad).
	Very stable.
	- MNIST(Converge)
	- CIFAR10(Converge)
	- Pokemon(Converge)


2017/7/29
--
It seems the WGAN has very hard generization compare with RBM.
	1. Using the VGCC 16 as initalization weights.

2017/7/31
--
	1. Colecting more data


EBGAN
---



Causal inference & minimize the total Risk
---
Structure Causal Model
	- W = fw(Uw)
	- A = fa(Ua)
	- Y = fy(W, A, Uy)


Generate Similar Pokemon
---
1. Reconstruction Loss.
2. Pokemon Loss(Reverse Problem).



Demo
---
1. Generate Wallpaper


Reference
---
[Gradient Decent](https://medium.com/intuitionmachine/the-peculiar-behavior-of-deep-learning-loss-surfaces-330cb741ec17)
