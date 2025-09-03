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
| Wed 9/3  | **Introduction**: Basics of univariate statistics | [Lecture 1](Lectures/Lecture01/3dasm_Lecture1.ipynb) | [Slides](Lectures/Lecture01/3dasm_Lecture1_slides.pdf) | [HW1 assigned](Assignments/3dasm_Homework1.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture01/3dasm_Lecture1.ipynb) |
| Fri 9/5  | **Practical tutorial**: Handling data with Pandas | [Lecture 2](Lectures/Lecture02/3dasm_Lecture2.ipynb) | [Slides](Lectures/Lecture02/3dasm_Lecture2_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture02/3dasm_Lecture2.ipynb) |
| Mon 9/8  | **Bayes' rule**: joint & conditional distributions | [Lecture 3](Lectures/Lecture03/3dasm_Lecture3.ipynb) | [Slides](Lectures/Lecture03/3dasm_Lecture3_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture03/3dasm_Lecture3.ipynb) |
| Wed 9/10  | **Multivariate statistics**: visualization of joint & conditional distributions | [Lecture 4](Lectures/Lecture04/3dasm_Lecture4.ipynb) | [Slides](Lectures/Lecture04/3dasm_Lecture4_slides.pdf) | ```HW1 due``` & [HW2 assigned](Assignments/3dasm_Homework2.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture04/3dasm_Lecture4.ipynb) |
| Fri 9/12  | **Bayesian inference for one hidden rv**: Example with Gaussian likelihood and Uniform prior (Part I) | [Lecture 5](Lectures/Lecture05/3dasm_Lecture5.ipynb) | [Slides](Lectures/Lecture05/3dasm_Lecture5_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture05/3dasm_Lecture5.ipynb) |
| Mon 9/15  | **Bayesian inference for one hidden rv**: Example with Gaussian likelihood and Uniform prior (Part II) | [Lecture 6](Lectures/Lecture06/3dasm_Lecture6.ipynb) | [Slides](Lectures/Lecture06/3dasm_Lecture6_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture06/3dasm_Lecture6.ipynb) |
| Wed 9/17  | **Bayesian inference for one hidden rv**: Redo example but now with Gaussian prior | [Lecture 7](Lectures/Lecture07/3dasm_Lecture7.ipynb) |  [Slides](Lectures/Lecture07/3dasm_Lecture7_slides.pdf) | ```HW2 due``` & [HW3 assigned](Assignments/3dasm_Homework3.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture07/3dasm_Lecture7.ipynb) |
| Fri 9/19  | **Machine Learning without going Bayesian**: Point Estimates | [Lecture 8](Lectures/Lecture08/3dasm_Lecture8.ipynb) | [Slides](Lectures/Lecture08/3dasm_Lecture8_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture08/3dasm_Lecture8.ipynb) |
| Mon 9/22  | **Linear Regression**: Practical tutorial (Part I: noiseless 1D example; underfitting vs. overfitting; interpolation vs. extrapolation) | [Lecture 9](Lectures/Lecture09/3dasm_Lecture9.ipynb) | [Slides](Lectures/Lecture09/3dasm_Lecture9_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture09/3dasm_Lecture9.ipynb) |
| Wed 9/24  | **Linear Regression**: Practical tutorial (Part II: noiseless vs. noisy datasets; train/test dataset split; multi-dimensional example) | [Lecture 10](Lectures/Lecture10/3dasm_Lecture10.ipynb) | [Slides](Lectures/Lecture10/3dasm_Lecture10_slides.pdf) | ```HW3 due``` & [HW4 assigned](Assignments/3dasm_Homework4.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture10/3dasm_Lecture10.ipynb) |
| Fri 9/26  | **Linear Regression**: Linear Least Squares model (Gaussian likelihood, Uniform prior, and Point Estimate) | [Lecture 11](Lectures/Lecture11/3dasm_Lecture11.ipynb) | [Slides](Lectures/Lecture11/3dasm_Lecture11_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture11/3dasm_Lecture11.ipynb) |
| Mon 9/29  | **Linear Regression**: Ridge, Lasso and Bayesian linear regression models (different likelihoods, priors and posteriors) | [Lecture 12](Lectures/Lecture12/3dasm_Lecture12.ipynb)  | [Slides](Lectures/Lecture12/3dasm_Lecture12_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture12/3dasm_Lecture12.ipynb) |
| Wed 10/1  | **Gaussian process regression**: theory | [Lecture 13](Lectures/Lecture13/3dasm_Lecture13.ipynb) | [Slides](Lectures/Lecture13/3dasm_Lecture13_slides.pdf) | ```HW4 due``` & [HW5 assigned](Assignments/3dasm_Homework5.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture13/3dasm_Lecture13.ipynb) |
| Fri 10/3  | **Gaussian process regression**: One dimensional tutorial (Part I: noiseless) | [Lecture 14](Lectures/Lecture14/3dasm_Lecture14.ipynb) | [Slides](Lectures/Lecture14/3dasm_Lecture14_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture14/3dasm_Lecture14.ipynb) |
| Mon 10/6  | **Gaussian process regression**: One dimensional tutorial (Part II: noisy) | [Lecture 14 (continued)](Lectures/Lecture14/3dasm_Lecture14.ipynb) | [Slides](Lectures/Lecture14/3dasm_Lecture14_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture14/3dasm_Lecture14.ipynb) |
| Wed 10/8  | **Gaussian process regression**: Multidimensional tutorial & importance of dataset scaling | [Lecture 15](Lectures/Lecture15/3dasm_Lecture15.ipynb) | [Slides](Lectures/Lecture15/3dasm_Lecture15_slides.pdf) | ```HW5 due``` & [HW6 assigned](Assignments/3dasm_Homework6.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture15/3dasm_Lecture15.ipynb) |
| Fri 10/10  | **Bayesian model selection & Hyperparameter optimization** |  [Lecture 16](Lectures/Lecture16/3dasm_Lecture16.ipynb) | [Slides](Lectures/Lecture16/3dasm_Lecture16_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture16/3dasm_Lecture16.ipynb) |
| Mon 10/13  | HOLIDAY 🥹 |  |  |  |
| Wed 10/15  | Q&A session |  |  | ```HW6 due``` |  |
| Fri 10/17  | **```MIDTERM Exam```** 🦾 |  |  |  |
| Mon 10/20  | **Framework for Data-Driven Design & Analysis of Structures & Materials**: [f3dasm](https://f3dasm.readthedocs.io/en/latest/) | [Lecture 17](Lectures/Lecture17/3dasm_Lecture17.ipynb) | [Slides](Lectures/Lecture17/3dasm_Lecture17_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture17/3dasm_Lecture17.ipynb) |
| Wed 10/22  | **[f3dasm](https://f3dasm.readthedocs.io/en/latest/) tutorial**: Data-driven process; Sampling methods; Simple model selection example |  [Lecture 18](Lectures/Lecture18/3dasm_Lecture18.ipynb) | [Slides](Lectures/Lecture18/3dasm_Lecture18_slides.pdf) | [HW7 assigned](Assignments/3dasm_Homework7.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture20/3dasm_Lecture20.ipynb) |
| Fri 10/24  | **[f3dasm](https://f3dasm.readthedocs.io/en/latest/) tutorial**: General use case (object oriented) | [Lecture 19](Lectures/Lecture19/3dasm_Lecture19.ipynb) | [Slides](Lectures/Lecture19/3dasm_Lecture19_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture19/3dasm_Lecture19.ipynb) |
| Mon 10/27  | **```Final Project Overview & Assignment```** 🦾  |  [Lecture 20](Lectures/Lecture20/3dasm_Lecture20.ipynb) | [Slides](Lectures/Lecture20/3dasm_Lecture20_slides.pdf) | [Final Project assigned](Assignments/3dasm_FinalProject.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture20/3dasm_Lecture20.ipynb) |
| Wed 10/29  | **Introduction to classification**: Tutorial with 3 simple classifiers on Iris dataset | [Lecture 21](Lectures/Lecture21/3dasm_Lecture21.ipynb) | [Slides](Lectures/Lecture21/3dasm_Lecture21_slides.pdf) | ```HW7 due``` & [HW8 assigned](Assignments/3dasm_Homework8.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture21/3dasm_Lecture21.ipynb) |
| Fri 10/31  | **Logistic regression classifier**: Classification as a regression task; Bernoulli observation distribution; the sigmoid trick | [Lecture 22](Lectures/Lecture22/3dasm_Lecture22.ipynb) | [Slides](Lectures/Lecture22/3dasm_Lecture22_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture22/3dasm_Lecture22.ipynb) |
| Mon 11/3  | **Logistic regression classifier**: Point estimate (e.g. MLE) as nonlinear optimization; classification as the mode of PPD | [Lecture 22 (continued)](Lectures/Lecture22/3dasm_Lecture22.ipynb) | [Slides](Lectures/Lecture22/3dasm_Lecture22_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture22/3dasm_Lecture22.ipynb) |
| Wed 11/5  | **Gaussian discriminant analysis (GDA) classifier**: Classification by a *generative* model | [Lecture 23](Lectures/Lecture23/3dasm_Lecture23.ipynb) | [Slides](Lectures/Lecture23/3dasm_Lecture23_slides.pdf) | ```HW8 due``` & [HW9 assigned](Assignments/3dasm_Homework9.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture23/3dasm_Lecture23.ipynb) |
| Fri 11/7  | **Optimization**: Part I | [Lecture 24](Lectures/Lecture24/3dasm_Lecture24.ipynb) | [Slides](Lectures/Lecture24/3dasm_Lecture24_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture24/3dasm_Lecture24.ipynb) |
| Mon 11/10  | **Optimization**: Part II | [Lecture 25](Lectures/Lecture25/3dasm_Lecture25.ipynb) | [Slides](Lectures/Lecture25/3dasm_Lecture25_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture25/3dasm_Lecture25.ipynb) |
| Wed 11/12  | **Optimization**: Part III | [Lecture 26](Lectures/Lecture26/3dasm_Lecture26.ipynb) | [Slides](Lectures/Lecture26/3dasm_Lecture26_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture26/3dasm_Lecture26.ipynb) |
| Fri 11/14  | **Artificial Neural Networks**: Part I | [Lecture 27](Lectures/Lecture27/3dasm_Lecture27.ipynb) | [Slides](Lectures/Lecture27/3dasm_Lecture27_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture27/3dasm_Lecture27.ipynb) |
| Mon 11/17  | **Artificial Neural Networks**: Part II | [Lecture 28](Lectures/Lecture28/3dasm_Lecture28.ipynb) | [Slides](Lectures/Lecture28/3dasm_Lecture28_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture28/3dasm_Lecture28.ipynb) |
| Wed 11/19  | **Artificial Neural Networks**: Part III | [Lecture 29](Lectures/Lecture29/3dasm_Lecture29.ipynb) | [Slides](Lectures/Lecture29/3dasm_Lecture29_slides.pdf) | ```HW9 due``` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture29/3dasm_Lecture29.ipynb) |
| Fri 11/21  | **Artificial Neural Networks**: Part IV | [Lecture 30](Lectures/Lecture30/3dasm_Lecture30.ipynb) | [Slides](Lectures/Lecture30/3dasm_Lecture30_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture30/3dasm_Lecture30.ipynb) |
| Mon 11/24  | **Thanksgiving week** 🦃 |  |  |  |  |
| Wed 11/26  | **Thanksgiving week** 🦃 |  |  |  |  |
| Fri 11/28  | **Thanksgiving week** 🦃 |  |  |  |  |
| Mon 12/01  | TBD | [Lecture 31](Lectures/Lecture31/3dasm_Lecture31.ipynb) | [Slides](Lectures/Lecture31/3dasm_Lecture31_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture31/3dasm_Lecture31.ipynb) |
| Wed 12/03  | TBD | [Lecture 32](Lectures/Lecture32/3dasm_Lecture32.ipynb) | [Slides](Lectures/Lecture32/3dasm_Lecture32_slides.pdf) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture32/3dasm_Lecture32.ipynb) |
| Fri 12/05  | **```Final Project presentations```** 🦾 |  |  | ```Final Project report due``` |  |


## Installation instructions

### **OPTION 1**. Run this notebook **locally in your computer**:

[Homework 1](Assignments/3dasm_Homework1.pdf) contains detailed instructions to install the virtual environment with all the packages required for this course. Below are more concise instructions for people familiar with installing mamba and tensorflow:

1. Install Mamba as described [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). (See Homework 1 for additional instructions)

2. Install [git](https://github.com/git-guides/install-git), open command window & clone the repository to your computer:
```
git clone https://github.com/bessagroup/3dasm_course
```

3. Create a virtual environment for this course called '3dasm':
```
mamba create -n 3dasm python==3.11 notebook nb_conda rise numpy scipy matplotlib pandas scikit-learn ipykernel ipywidgets
```

4. Activate the '3dasm' virtual environment:

```
mamba activate 3dasm
```

5. Install [pytorch](https://pytorch.org/) in the '3dasm' virtual environment. (See Homework 1 for additional instructions)

```
pip install torch torchvision –index-url https://download.pytorch.org/whl/cu126
```

6. Install f3dasm package:
```
pip install f3dasm==2.0.2
```

7. Install optuna package:
```
pip install optuna
```

After you installed every package, you are ready to go! Reboot your computer.

- Open a (mamba) command window, activate the '3dasm' environment and start jupyter notebook (it will open in your internet browser):
```
mamba activate 3dasm
jupyter notebook
``` 
- Open notebook (for example: *3dasm_course/Lectures/Lecture01/3dasm_Lecture1.ipynb*).

You're all set!

### **OPTION 2**. Use **Google's Colab** (no installation required, but Colab session times out if idle):

1. go to [Google Colab](https://colab.research.google.com)
2. Login with your credentials
3. File > Open notebook
4. Click on Github (no need to login or authorize anything)
5. paste the git link: https://github.com/bessagroup/3dasm_course
6. click search and then click on the notebook (for example: *3dasm_course/Lectures/Lecture01/3dasm_Lecture1.ipynb*)
