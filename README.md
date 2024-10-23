# Data-Driven Design &amp; Analysis of Structures &amp; Materials (3dasm)

Miguel A. Bessa | miguel_bessa@brown.edu | Associate Professor

## Introduction

**What:** This course aims to be an introduction to machine learning from a probabilistic perspective.

**Where:** This notebook comes from this [repository](https://github.com/bessagroup/3dasm_course)

**Reference:** Murphy, Kevin P. *Probabilistic machine learning: an introduction*. MIT press, 2022. Available online [here](https://probml.github.io/pml-book/book1.html)

**How:** We try to follow Murphy's book closely, but the sequence of Chapters and Sections is different. The intention is to use notebooks as an introduction to the topic and Murphy's book as a resource.
* If working offline: Go through this notebook and read the book.
* If attending class in person: listen to me (!) but also go through the notebook in your laptop at the same time. Read the book.
* If attending lectures remotely: listen to me (!) via Zoom and (ideally) use two screens where you have the notebook open in 1 screen and you see the lectures on the other. Read the book.

**Folder structure**
* The "Lectures" folder contains each lecture in a separate folder "LectureX" where X is the lecture number.
* Each "LectureX" folder contains:
    1. A jupyter notebook "3dasm_LectureX.ipynb" that you can run locally or in servers like Google Colab.
    2. A pdf "3dasm_LectureX slides.pdf" with the slides of that lecture.
    4. A "your_data" folder that you can use to create data or other things in your own computer.
* The preferred method to follow the course is to look directly into the jupyter notebook, as it contains additional notes and working code.

**Grading**

Homeworks 30%, Midterm 30%, and Final Project 40%.

Homeworks will be graded only with 5 levels: A+ (100%; fully correct), A (90%; has minor error), B (75%; has significant error), C (60%; mostly incorrect but homework was delivered), D (0%, not delivered). If you deliver something with an honest attempt at solving the homework you get 60% for that homework.


> [!NOTE]
> Late Homework can only get up to A (90%).
> 
> The worst Homework is removed from the final grade.

**Course outline**

| DATE | SUBJECT | Notebook | PDF | Homework | Colab |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Wed 9/4  | **Introduction**: Basics of univariate statistics | [Lecture 1](Lectures/Lecture01/3dasm_Lecture1.ipynb) | [Slides](Lectures/Lecture01/3dasm_Lecture1_slides.pdf) | [HW1 assigned](Assignments/3dasm_Homework1.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture01/3dasm_Lecture1.ipynb) |
| Fri 9/6  | **Practical tutorial**: Handling data with Pandas | [Lecture 2](Lectures/Lecture02/3dasm_Lecture02.ipynb) | [Slides](Lectures/Lecture02/3dasm_Lecture2_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture02/3dasm_Lecture2.ipynb) |
| Mon 9/9  | **Bayes' rule**: joint & conditional distributions | [Lecture 3](Lectures/Lecture03/3dasm_Lecture3.ipynb) | [Slides](Lectures/Lecture03/3dasm_Lecture3_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture03/3dasm_Lecture3.ipynb) |
| Wed 9/11  | **Multivariate statistics**: visualization of joint & conditional distributions | [Lecture 4](Lectures/Lecture04/3dasm_Lecture4.ipynb) | [Slides](Lectures/Lecture04/3dasm_Lecture4_slides.pdf) | ```HW1 due``` & [HW2 assigned](Assignments/3dasm_Homework2.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture04/3dasm_Lecture4.ipynb) |
| Fri 9/13  | **Bayesian inference for one hidden rv**: Example with Gaussian likelihood and Uniform prior (Part I) | [Lecture 5](Lectures/Lecture05/3dasm_Lecture5.ipynb) | [Slides](Lectures/Lecture05/3dasm_Lecture5_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture05/3dasm_Lecture5.ipynb) |
| Mon 9/16  | **Bayesian inference for one hidden rv**: Example with Gaussian likelihood and Uniform prior (Part II) | [Lecture 6](Lectures/Lecture06/3dasm_Lecture6.ipynb) | [Slides](Lectures/Lecture06/3dasm_Lecture6_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture06/3dasm_Lecture6.ipynb) |
| Wed 9/18  | **Bayesian inference for one hidden rv**: Redo example but now with Gaussian prior | [Lecture 7](Lectures/Lecture07/3dasm_Lecture7.ipynb) |  [Slides](Lectures/Lecture07/3dasm_Lecture7_slides.pdf) | ```HW2 due``` & [HW3 assigned](Assignments/3dasm_Homework3.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture07/3dasm_Lecture7.ipynb) |
| Fri 9/20  | **Machine Learning without going Bayesian**: Point Estimates | [Lecture 8](Lectures/Lecture08/3dasm_Lecture8.ipynb) | [Slides](Lectures/Lecture08/3dasm_Lecture8_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture08/3dasm_Lecture8.ipynb) |
| Mon 9/23  | **Linear Regression**: Practical tutorial (Part I: noiseless 1D example; underfitting vs. overfitting; interpolation vs. extrapolation) | [Lecture 9](Lectures/Lecture09/3dasm_Lecture9.ipynb) | [Slides](Lectures/Lecture09/3dasm_Lecture9_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture09/3dasm_Lecture9.ipynb) |
| Wed 9/25  | **Linear Regression**: Practical tutorial (Part II: noiseless vs. noisy datasets; train/test dataset split; multi-dimensional example) | [Lecture 10](Lectures/Lecture10/3dasm_Lecture10.ipynb) | [Slides](Lectures/Lecture10/3dasm_Lecture10_slides.pdf) | ```HW3 due``` & [HW4 assigned](Assignments/3dasm_Homework4.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture10/3dasm_Lecture10.ipynb) |
| Fri 9/27  | **Linear Regression**: Linear Least Squares model (Gaussian likelihood, Uniform prior, and Point Estimate) | [Lecture 11](Lectures/Lecture11/3dasm_Lecture11.ipynb) | [Slides](Lectures/Lecture11/3dasm_Lecture11_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture11/3dasm_Lecture11.ipynb) |
| Mon 9/30  | **Linear Regression**: Ridge, Lasso and Bayesian linear regression models (different likelihoods, priors and posteriors) | [Lecture 12](Lectures/Lecture12/3dasm_Lecture12.ipynb)  | [Slides](Lectures/Lecture12/3dasm_Lecture12_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture12/3dasm_Lecture12.ipynb) |
| Wed 10/2  | **Gaussian process regression**: theory | [Lecture 13](Lectures/Lecture13/3dasm_Lecture13.ipynb) | [Slides](Lectures/Lecture13/3dasm_Lecture13_slides.pdf) | ```HW4 due``` & [HW5 assigned](Assignments/3dasm_Homework5.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture13/3dasm_Lecture13.ipynb) |
| Fri 10/4  | **Gaussian process regression**: One dimensional tutorial (Part I: noiseless) | [Lecture 14](Lectures/Lecture14/3dasm_Lecture14.ipynb) | [Slides](Lectures/Lecture14/3dasm_Lecture14_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture14/3dasm_Lecture14.ipynb) |
| Mon 10/7  | **Gaussian process regression**: One dimensional tutorial (Part II: noisy) | [Lecture 14 (continued)](Lectures/Lecture14/3dasm_Lecture14.ipynb) | [Slides](Lectures/Lecture14/3dasm_Lecture14_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture14/3dasm_Lecture14.ipynb) |
| Wed 10/9  | **Gaussian process regression**: Multidimensional tutorial & importance of dataset scaling | [Lecture 15](Lectures/Lecture15/3dasm_Lecture15.ipynb) | [Slides](Lectures/Lecture15/3dasm_Lecture15_slides.pdf) | ```HW5 due``` & [HW6 assigned](Assignments/3dasm_Homework6.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture15/3dasm_Lecture15.ipynb) |
| Fri 10/11  | **Bayesian model selection & Hyperparameter optimization** |  [Lecture 16](Lectures/Lecture16/3dasm_Lecture16.ipynb) | [Slides](Lectures/Lecture16/3dasm_Lecture16_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture16/3dasm_Lecture16.ipynb) |
| Mon 10/14  | HOLIDAY 🥹 |  |  |  |
| Wed 10/16  | Q&A session |  |  | ```HW6 due``` |  |
| Fri 10/18  | **```MIDTERM Exam```** 🦾 |  |  |  |
| Mon 10/21  | **Framework for Data-Driven Design & Analysis of Structures & Materials**: [f3dasm](https://f3dasm.readthedocs.io/en/latest/) | [Lecture 17](Lectures/Lecture17/3dasm_Lecture17.ipynb) | [Slides](Lectures/Lecture17/3dasm_Lecture17_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture17/3dasm_Lecture17.ipynb) |
| Wed 10/23  | **[f3dasm](https://f3dasm.readthedocs.io/en/latest/) tutorial**: Data-driven process; Sampling methods; Simple model selection example |  [Lecture 18](Lectures/Lecture18/3dasm_Lecture18.ipynb) | [Slides](Lectures/Lecture18/3dasm_Lecture18_slides.pdf) | [HW7 assigned](Assignments/3dasm_Homework7.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture20/3dasm_Lecture20.ipynb) |
| Fri 10/25  | **[f3dasm](https://f3dasm.readthedocs.io/en/latest/) tutorial**: General use case (object oriented) | [Lecture 19](Lectures/Lecture19/3dasm_Lecture19.ipynb) | [Slides](Lectures/Lecture19/3dasm_Lecture19_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture19/3dasm_Lecture19.ipynb) |
| Mon 10/28  | **```Final Project Overview & Assignment```** 🦾  |  [Lecture 20](Lectures/Lecture20/3dasm_Lecture20.ipynb) | [Slides](Lectures/Lecture20/3dasm_Lecture20_slides.pdf) | [Final Project assigned](Assignments/3dasm_FinalProject.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture20/3dasm_Lecture20.ipynb) |
| Wed 10/30  | **Introduction to classification**: tutorial with 3 simple classifiers on Iris dataset | [Lecture 21](Lectures/Lecture21/3dasm_Lecture21.ipynb) | [Slides](Lectures/Lecture21/3dasm_Lecture21_slides.pdf) | ```HW7 due``` & [HW8 assigned](Assignments/3dasm_Homework8.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture21/3dasm_Lecture21.ipynb) |
| Fri 11/1  | **Logistic regression classifier**: classification as a regression task | [Lecture 22](Lectures/Lecture22/3dasm_Lecture22.ipynb) | [Slides](Lectures/Lecture22/3dasm_Lecture22_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture22/3dasm_Lecture22.ipynb) |
| Mon 11/4  | **Gaussian discriminant analysis (GDA) classifier**: Classification by a *generative* model | [Lecture 23](Lectures/Lecture23/3dasm_Lecture23.ipynb) | [Slides](Lectures/Lecture23/3dasm_Lecture23_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture23/3dasm_Lecture23.ipynb) |
| Wed 11/6  | **Optimization**: Part I | [Lecture 24](Lectures/Lecture24/3dasm_Lecture24.ipynb) | [Slides](Lectures/Lecture24/3dasm_Lecture24_slides.pdf) | ```HW8 due``` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture24/3dasm_Lecture24.ipynb) |
| Fri 11/8  | **Optimization**: Part II | [Lecture 25](Lectures/Lecture25/3dasm_Lecture25.ipynb) | [Slides](Lectures/Lecture25/3dasm_Lecture25_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture25/3dasm_Lecture25.ipynb) |
| Mon 11/11  | TBD | [Lecture 26](Lectures/Lecture26/3dasm_Lecture26.ipynb) | [Slides](Lectures/Lecture26/3dasm_Lecture26_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture26/3dasm_Lecture26.ipynb) |
| Wed 11/13  | TBD | [Lecture 27](Lectures/Lecture27/3dasm_Lecture27.ipynb) | [Slides](Lectures/Lecture27/3dasm_Lecture27_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture27/3dasm_Lecture27.ipynb) |
| Fri 11/15  | TBD | [Lecture 28](Lectures/Lecture28/3dasm_Lecture28.ipynb) | [Slides](Lectures/Lecture28/3dasm_Lecture28_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture28/3dasm_Lecture28.ipynb) |
| Mon 11/18  | TBD | [Lecture 29](Lectures/Lecture29/3dasm_Lecture29.ipynb) | [Slides](Lectures/Lecture29/3dasm_Lecture29_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture29/3dasm_Lecture29.ipynb) |
| Wed 11/20  | TBD | [Lecture 30](Lectures/Lecture30/3dasm_Lecture30.ipynb) | [Slides](Lectures/Lecture30/3dasm_Lecture30_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture30/3dasm_Lecture30.ipynb) |
| Fri 11/22  | TBD | [Lecture 31](Lectures/Lecture31/3dasm_Lecture31.ipynb) | [Slides](Lectures/Lecture31/3dasm_Lecture31_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture31/3dasm_Lecture31.ipynb) |
| Mon 11/25  | **Thanksgiving week** 🦃 |  |  |  |  |
| Wed 11/27  | **Thanksgiving week** 🦃 |  |  |  |  |
| Fri 11/29  | **Thanksgiving week** 🦃 |  |  |  |  |
| Mon 12/02  | TBD | [Lecture 32](Lectures/Lecture32/3dasm_Lecture32.ipynb) | [Slides](Lectures/Lecture32/3dasm_Lecture32_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture32/3dasm_Lecture32.ipynb) |
| Wed 12/04  | TBD | [Lecture 33](Lectures/Lecture33/3dasm_Lecture33.ipynb) | [Slides](Lectures/Lecture33/3dasm_Lecture33_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture33/3dasm_Lecture33.ipynb) |
| Fri 12/06  | **```Final Project presentations```** 🦾 |  |  | ```Final Project report due``` |  |


## Installation instructions

### **OPTION 1**. Run this notebook **locally in your computer**:

[Homework 1](Assignments/3dasm_Homework1.pdf) contains detailed instructions to install the virtual environment with all the packages required for this course. Below are more concise instructions for people familiar with installing mamba and tensorflow:

1. Install Mamba as described [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). (See Homework 1 for additional instructions)
2. Install Jupyter notebook and extensions in base environment
```
mamba install -c anaconda notebook nb_conda rise
```

3. Create a virtual enviroment for this course called '3dasm':

```
mamba create -n 3dasm python==3.10 numpy scipy matplotlib pandas scikit-learn ipykernel ipywidgets f3dasm
```

4. Install [git](https://github.com/git-guides/install-git), open command window & clone the repository to your computer:
```
git clone https://github.com/bessagroup/3dasm_course
```

5. Install [tensorflow](https://www.tensorflow.org/install/pip) in the '3dasm' virtual environment. (See Homework 1 for additional instructions)

6. Install scikeras in '3dasm' virtual environment:
```
mamba activate 3dasm
pip install scikeras
```

After you installed every package, you are ready to go!

- Open a (mamba) command window and load jupyter notebook (it will open in your internet browser):
```
jupyter notebook
``` 
- Open notebook (for example: *3dasm_course/Lectures/Lecture1/3dasm_Lecture1.ipynb*) and choose the '3dasm' kernel.

You're all set!

### **OPTION 2**. Use **Google's Colab** (no installation required, but times out if idle):

1. go to [Google Colab](https://colab.research.google.com)
2. Login with your credentials
3. File > Open notebook
4. Click on Github (no need to login or authorize anything)
5. paste the git link: https://github.com/bessagroup/3dasm_course
6. click search and then click on the notebook (for example: *3dasm_course/Lectures/Lecture1/3dasm_Lecture1.ipynb*)
