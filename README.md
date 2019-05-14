
<p align="center"> Ukrainian Catholic University  </p>

<p align="center"> Faculty of Applied Sciences </p>

<p align="center"> Data Science Master Programme</p>

<p align="center"> 12 May 2019 </p>

<p align="center"><b> <big>Responsible Data Science final project</big></b> </p>

<p align="right" ><b><i>Authors:</i></b></p> 
<p align="right"> Anastasiia Khaburska</p>
<p align="right"> Anton Shcherbyna</p>
<p align="right"> Vadym Korshunov</p>
<p align="right"> Yaroslava Lochman</p>

## Interpretability of Fake Face Discrimination
The project idea was inspired by Kaggle competition ["Real and Fake Face Detection"](https://www.kaggle.com/ciplab/real-and-fake-face-detection) organised by Department of Computer Science, Yonsei University. 

In this work, we took StyleGAN - pretrained generator of fake faces - and StarGAN - pretrained discriminator of faces - and investigated behaviour of this discriminative CNN model for the **Transparency** and **Interpretability** goals. We interpreted the model's output on the fake generated images. To accomplish this task, we  explored the most popular feature visualisation techniques and tried a few of them implemented in Pytorch:

-  Vanilla Backpropagation (with Smooth Gradient)
-  Guided Backpropagation
-  Gradient-weighted class activation mapping ([GradCAM](https://arxiv.org/pdf/1610.02391.pdf))

 :notebook: **This [Colab Notebook](https://colab.research.google.com/drive/1I9bWp_wwu3kui8-AC6ghK6aikyE5XBIo
) is an interactive Report on our project.** 

All the code, we where working on, may be seen our [GitHub repository](https://github.com/Anastasiia-Khab/responsible-ds-final-project)

Also, a great help for us in understanding and implementing of CNN visualisation techniques 
was this GitHub repository with [PyTorch CNN Visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
