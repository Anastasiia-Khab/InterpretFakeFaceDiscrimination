
<p align="center">

Ukrainian Catholic University  

Faculty of Applied Sciences

Data Science Master Programme

12 May 2019

## *Interpretability of Fake Face Discrimination*
## Responsible Data Science final project  

*Authors:*
- Anastasiia Khaburska
- Anton Shcherbyna
- Vadym Korshunov
- Yaroslava Lochman
</p>

The project idea was inspired by Kaggle competition ["Real and Fake Face Detection"](https://www.kaggle.com/ciplab/real-and-fake-face-detection) organised by Department of Computer Science, Yonsei University. 

In this work, we took StyleGAN - pretrained generator of fake faces - and StarGAN - pretrained discriminator of faces - and investigated behaviour of this discriminative CNN model for the **Transparency** and **Interpretability** goals. We interpreted the model's output on the fake generated images. To accomplish this task, we  explored the most popular feature visualisation techniques and tried a few of them implemented in Pytorch:

1.  Vanilla Backpropagation (with Smooth Gradient)
2.  Guided Backpropagation
3.  Gradient-weighted class activation mapping ([GradCAM](https://arxiv.org/pdf/1610.02391.pdf))

**This Colab Notebook is an interactive Report on our project:** https://colab.research.google.com/drive/1I9bWp_wwu3kui8-AC6ghK6aikyE5XBIo

All the code, we where working on, may be seen our [GitHub repository](https://github.com/Anastasiia-Khab/responsible-ds-final-project)

Also, a great help for us in understanding and implementing of CNN visualisation techniques 
was this GitHub repository with [PyTorch CNN Visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
