# Getting started with NNs

![](neural_net_img.png)

This guide is for anyone who wants to start working with neural networks but has little to no prior experience and does not know where to start. It can useful if you are a math student, a life scientist or anyone else interested in data analysis. 
We will cover basic concepts, as well as programming tools, that you need to get started with neural networks. The guide is organized into sections as in the map above and you can read it in the order that is most convenient for you, as well as skip some sections altogether if you are familiar with concepts covered in them.

![](mindmap.png)

# Table of Contents
1. [Math Background](#math_background) 
2. [Neural Network Basics](#nns_basics)
3. [Command Line Basics](#cl_basics)
5. [Hardware and OS](#hard_os)
4. [Python](#python)
5. [Deep Learning Frameworks](#dl_frameworks)
6. [Training Neural Nets](#train_nns)
7. [Research Experiments with neural nets](#resexp_nns)
8. [Data Analysis](#data_an)

## Math Background <a name="math_background"></a>

Areas of math that are most commonly used in neural networks are (more or less in the order of importance):

* **Linear Algebra** (vectors, matrices and various operatins with them)
* **Calculus** (gradients and integrals)
* **Probability and Statistics** (random variables, expectations, variance, Bayes' theorem)
* **Optimization Algorithms** (minimization or maximization)
 
If you are familiar with basics of these fields, especially with matrices and gradients (e.g. you took Linear Algebra and Multivariable Calculus courses), you can skip this section. Otherwise we recommend that you obtain basic understanding of matrices and gradients. 

You can start with videos by [3blue1brown](https://www.3blue1brown.com), who created a series on the basics of linear algebra, calculus and multivariable calculus. You can either watch the entire series or just stop watching after feeling comfortable with matrices and gradients. 

* [Linear Algebra Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
* [Calculus Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
* [Multivariable Calculus Series](https://www.youtube.com/watch?v=TrcCbdWwCBc&list=PLSQl0a2vh4HC5feHa6Rc5c0wbRTx56nF7)

You can also have a look at the following free online courses from [Khan academy](https://www.khanacademy.org):

* [Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
* [Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus)
* [Statistics and Probability](https://www.khanacademy.org/math/statistics-probability)

If you prefer written or more in-depth sources, you can have a look at the following:

* [_Linear Algebra Review and Reference_ by Zico Kolter (updated by Chuong Do)](http://cs229.stanford.edu/section/cs229-linalg.pdf)
* [_Introduction to Linear Algebra_ by Gilbert Strang](http://math.mit.edu/~gs/linearalgebra/)
* [_Review of Probability Theory_ by Arian Maleki and Tom Do](http://cs229.stanford.edu/section/cs229-prob.pdf)

For the next section you should understand why the gradient is pointing in the direction of steepest descent and  matrix multiplication.

## Neural Network Basics <a name="nn_basics"></a>
You can skip this section if you already know what a neural network is, what a loss function and the backpropagation algroithm is. Before traininig neural networks you should know what a neural network is. The following video series by [3blue1brown](https://www.3blue1brown.com) provides an excellent intuitive introduction to the basics of neural networks.

[https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)


## Command Line Basics<a name="cl_basics"></a>
To run code on your computer or work with remote machines, you often need to use command line. The command languages (or shells) that are available for you depend on your OS. The most commonly used one is Bash, which is the default for most Linux systems and MacOS prior to 2019. You can start learning about it with this tutorial: [https://ubuntu.com/tutorials/command-line-for-beginners#1-overview](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview)

Other common shells for Linux or MacOS (e.g. [Zsh](https://en.wikipedia.org/wiki/Z_shell)) are quite similar to Bash and allow easy transition.

On the other hand, Windows uses Command Prompt (cmd), which is rather different from Bash. You can take a look at this tutorial if you have interest on learning cmd: [https://www.cs.princeton.edu/courses/archive/spr05/cos126/cmd-prompt.html](https://www.cs.princeton.edu/courses/archive/spr05/cos126/cmd-prompt.html)

## Hardware and OS<a name="hard_os"></a>

The training of neural networks is usually computationally expensive. Modern deep learning frameworks have included the possibility to train the neural network in different architectures and devices, in particular, GPUs. Although the need of a GPU is generally minor when you are training your first experiments, a real-world application will sometimes need more than one GPU to be trained. The advantages of frameworks like [pytorch or tensorflow](#dl_framewokrs) is the compatibility of its code to almost any device. Since the software normally used in deep learning is python, one can train neural networks on any operating system, e.g. Windows, OS X and Linux. We recommend to use Windows or Linux since currently OS X does not offer a stable GPU version. 

In order to use GPUs for training you need to have a CUDA-compatible GPU from NVIDIA. If you have Linux Ubuntu, you can follow this [post](https://askubuntu.com/questions/1288672/how-do-you-install-cuda-11-on-ubuntu-20-10-and-verify-the-installation). For Windows 10 users, you can consult this [guide](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781). 

If you are affiliated to a university or a research center you would normally have access to computing clusters with GPUs. The main tool you need to know to run your training remotely is [ssh](https://www.hostinger.com/tutorials/ssh-tutorial-how-does-ssh-work). If you are interested to have your own Deep Learning Rig, there are affordable ways to build it; this [video](https://www.youtube.com/watch?v=Nz7xzUybpFM&ab_channel=DanielBourke) provides a detailed guide on how to do it. There are also some commercially available pre-built deep learning workstations (high-performance PCs), for example in Germany, there is [AIME](https://www.aime.info/).

## Python<a name="python"></a>
Python is a very popular programming language! It is fair to say that most of deep learning research and applications require python. The following chart illustrates the rise of popularity of python.
![](chart_python.png)
(Source: https://stackoverflow.blog/2017/09/06/incredible-growth-python/ )

There is an abundance of tutorials for python. We recommend 
[https://www.python-course.eu/python3_course.php](https://www.python-course.eu/python3_course.php)

 The main advantage of python is the available optimized libraries for scientific computing, for example, [numpy](https://numpy.org/doc/stable/) and [scipy](https://docs.scipy.org/doc/scipy/reference/). For visualiation [matplotlib](https://matplotlib.org/stable/contents.html) is typically used. We recommend anybody to create a local enviroment to install all your libraries wihtout affecting the global system. This can be done using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html), conda is also useful to install GPU-capable libraries. Within conda you can use [pip](https://www.anaconda.com/blog/using-pip-in-a-conda-environment) for easy installation of the libraries. 

In scientific computing, you typically would also like to interact with the results of your computations and visualize them in real-time. [Jupyter Notebook](https://jupyter.org/documentation) is the best tool to do that in Python. It allows you to visualize and run individual pieces in real-time, which is ideal for prototyping. Jupyter Notebook is also available for remote computation, one can learn to run notebooks remotely following this [guide](https://fizzylogic.nl/2017/11/06/edit-jupyter-notebooks-over-ssh/). This also allows you to have a graphical interface on remote servers. Another great visualiation tool similar to Jupyter is [Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true). As Jupyter, in Colab you can use interactive notebooks to run python code and train neural networks. The big advantage of this tool is that it makes all the computations in the cloud. This also means that you can run your notebook whenever you have internet access. Google also allows Colab users to train and deploy models, both freely and with cost, with GPUs and TPUs. 

## Deep Learning Frameworks <a name="dl_frameworks"></a>
There is a great number of python libraries that provide implementations of neural networks, but the most popular ones are Tensorflow and PyTorch:

* [PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)
* [Tensorflow](https://www.tensorflow.org/tutorials) (with its user-friendly [Keras](https://keras.io/about/) API)

Both of the libraries allow similar functionality and are well-documented. They are also compatible with a lot of architectures, such as CPU, GPU and TPU. The choice between them either depends on your project's needs or is just subjective. You can consult some _recent_ blogposts (e.g. [this one](https://medium.com/featurepreneur/tensorflow-vs-pytorch-which-is-better-for-your-application-development-6897d5d4dee0)) to make your choice.

## Training Neural Nets <a name="train_nns"></a>
There are numerous choices you have to make while building and training a neural network model. They can be categorized as follows:

**Architecture:** First you need to choose the very structure of a network. How many layers should it have? What kind of layers in what order? How many neurons/filters should be in each layer? The number of particular architectures published in deep learning research is enormous and it's impossible to cover all of them. But to understand more complex architectures, it is important to consider at least these basic classes:
 * Fully-connected networks (Multi-layer perceptrons)
 * Convolutional networks
 * Recurrent networks
 * Residual netoworks
 * Transformers

**Optimization method:** There is a number of optimization methods beyond gradient descent that are commonly used in deep learning and you need to choose one of them to train your network. You can find a good survey of optimization methods for deep learning in this [blogpost](https://medium.com/analytics-vidhya/different-optimization-algorithm-for-deep-neural-networks-complete-guide-7f3e49eb7d42).  Often adaptive optimization methods or methods with momentum yield better results than simple gradient descent and the Adam algorithm is a very popular choice. 

**Hyperparameters:** You need to understand and reasonably choose hyperparameters involved in training, such as learning rate, and batch size. This [post](https://neptune.ai/blog/hyperparameter-tuning-in-python-a-complete-guide-2020) is an up-to-date study of available hyperparameter tunning algorithms and implementations. 

**Initialization:** Initialization of your weight can make the difference for your network to converge succesfully to good minima. In this [article](https://www.deeplearning.ai/ai-notes/initialization/) there is a detailed discussion on the commonly used initialization procedures. 

**Layers:** A neural network architecture is defined by its basic components, the layers. The most commonly used layers are for example:
  * [Dense layer](https://medium.com/datathings/dense-layers-explained-in-a-simple-way-62fe1db0ed75)
  * [Convolution layers](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)
  * [Pooling layer](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)
  * [Batch Norm layer](https://www.youtube.com/watch?v=nUUqwaxLnWs)
  * [Recurrent layers](https://medium.com/datathings/recurrent-lstm-layers-explained-in-a-simple-way-d615ebcac450)
  * [Residual layers](https://medium.com/analytics-vidhya/understanding-and-implementation-of-residual-networks-resnets-b80f9a507b9c)
  * [Attention layers](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

These layers and many variations of them are implemented in the deep learning frameworks that we covered.

## Research Experiments with Neural Nets<a name="resexp_nns"></a>

Deep learning is a field that has an important empirical side. In order to train a neural network succesfully merely choosing the design is not enough, you would need to make trial-error iterations in order to tune the different elements. You can evaluate the performance of your neural network using different metrics, such as accuracy and mean square error. Software like [tensorboard](https://www.tensorflow.org/tensorboard?hl=es-419) allows you to monitor the performance of different runs simultaneously. Tools like [Keras Tunner](https://www.tensorflow.org/tutorials/keras/keras_tuner) in tensorflow and [Ray Tune](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html) in pytorch are handy for optimization and fine-tunning the different hyperparameters of your architecture. 

## Data Analysis and Image Processing <a name="data_an"></a>

Data Analysis and Image processing tools are also very handy in the training of neural networks. In python the main library used for data processing and analysis is [pandas](https://pandas.pydata.org/docs/), inspired by the statistical programming language [R](https://www.r-project.org/about.html). For image processing we recommend the library [sci-kit image](https://scikit-image.org/docs/stable/) which contains plenty of image filtering, resizing, cropping, rotating, etc... algorithms.
