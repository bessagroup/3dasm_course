# Data-Driven Design &amp; Analysis of Structures &amp; Materials (3dasm)

Miguel A. Bessa | <miguel_bessa@brown.edu> | Associate Professor

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
    2. A "your_data" folder that you can use to create data or other things in your own computer.
    3. The `Lectures/data` and `Lectures/figures` folders hold datasets and figures shared by several lectures.
* The preferred method to follow the course is to look directly into the jupyter notebook, as it contains additional notes and working code.

**Grading**

The [syllabus](Lectures/Syllabus/3dasm_syllabus.pdf) is the official course document; office hours and classroom are listed on the Canvas course page and on [Courses@Brown](https://cab.brown.edu/).

Homeworks 10%, Midterm 40%, and Final Project 50%.

Homeworks will be graded only with 5 levels: A+ (100%; fully correct), A (90%; has minor error), B (75%; has significant error or several minor errors), C (60%; several significant errors), D (0%, homework not delivered). If you deliver something with an honest attempt at solving the homework you get 60% for that homework.

> [!NOTE]
> Late Homework can only get up to A (90%).
>
> The worst Homework is removed from the final grade.

**Using LLMs and AI coding agents in this course**

You are encouraged to use large language models (ChatGPT, Claude, Gemini, open-weight models, etc.) and AI coding agents (Claude Code, opencode, Copilot, etc.) throughout this course, including for homework. However, be aware that **using an LLM to do your homework can significantly impair your learning**: two large controlled studies found that students with AI help scored much better on homework and then scored *worse* than students without AI on the exams that followed.
* Bastani et al. (2025), a randomized trial with about 1,000 high-school students: GPT-4 access raised practice performance by 48%, and those students then scored 17% *lower* than the control group on the exam. [*PNAS* 122(26)](https://www.pnas.org/doi/10.1073/pnas.2422633122)
* Strömberg, Lei & Wu (2026), 30 months of data on 26,811 secondary-school students: AI use raised homework scores by 18%; unaided exam scores fell by 20% within six months. 81% of AI users used it to generate answers; only those students were harmed. [CEPR Discussion Paper 21577](https://econpapers.repec.org/paper/cprceprdp/21577.htm)

This is why:
1. Homeworks only weigh 10% of the grade because good LLMs can solve them.
2. The midterm is closed-book (no computer, no Internet, and... no LLM).
3. The Final Project has an open-ended component. Working with an LLM is encouraged. The question is: how great can that project become?

Practical instructions for setting up an AI coding agent (with an Anthropic account, or with free open-weight models) will be given in class during the first weeks.

**Course outline**

| DATE | SUBJECT | Notebook | Reading | Homework | Colab |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Wed 9/9  | **Introduction**: Basics of univariate statistics | [Lecture 1](Lectures/Lecture01/3dasm_Lecture1.ipynb) | Chapter 2 until Section 2.3 | [HW1 assigned](Assignments/3dasm_Homework1.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture01/3dasm_Lecture1.ipynb) |
| Fri 9/11  | **Practical tutorial**: Handling data with Pandas | [Lecture 2](Lectures/Lecture02/3dasm_Lecture2.ipynb) |  |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture02/3dasm_Lecture2.ipynb) |
| Mon 9/14  | **Bayes' rule**: joint & conditional distributions | [Lecture 3](Lectures/Lecture03/3dasm_Lecture3.ipynb) | Chapter 3 (until Section 3.3) |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture03/3dasm_Lecture3.ipynb) |
| Wed 9/16  | **Multivariate statistics**: visualization of joint & conditional distributions | [Lecture 4](Lectures/Lecture04/3dasm_Lecture4.ipynb) | Bishop's book Section 2.3 | ```HW1 due``` & [HW2 assigned](Assignments/3dasm_Homework2.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture04/3dasm_Lecture4.ipynb) |
| Fri 9/18  | **Bayesian inference for one hidden rv**: Example with Gaussian likelihood and Uniform prior (Part I) | [Lecture 5](Lectures/Lecture05/3dasm_Lecture5.ipynb) | Chapter 3 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture05/3dasm_Lecture5.ipynb) |
| Mon 9/21  | **Bayesian inference for one hidden rv**: Example with Gaussian likelihood and Uniform prior (Part II) | [Lecture 6](Lectures/Lecture06/3dasm_Lecture6.ipynb) | Chapter 3 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture06/3dasm_Lecture6.ipynb) |
| Wed 9/23  | **Bayesian inference for one hidden rv**: Redo example but now with Gaussian prior | [Lecture 7](Lectures/Lecture07/3dasm_Lecture7.ipynb) |  | ```HW2 due``` & [HW3 assigned](Assignments/3dasm_Homework3.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture07/3dasm_Lecture7.ipynb) |
| Fri 9/25  | **Machine Learning without going Bayesian**: Point Estimates | [Lecture 8](Lectures/Lecture08/3dasm_Lecture8.ipynb) | Chapter 4 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture08/3dasm_Lecture8.ipynb) |
| Mon 9/28  | **Linear Regression**: Practical tutorial (Part I: noiseless 1D example; underfitting vs. overfitting; interpolation vs. extrapolation) | [Lecture 9](Lectures/Lecture09/3dasm_Lecture9.ipynb) | Chapter 11 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture09/3dasm_Lecture9.ipynb) |
| Wed 9/30  | **Linear Regression**: Practical tutorial (Part II: noiseless vs. noisy datasets; train/test dataset split; multi-dimensional example) | [Lecture 10](Lectures/Lecture10/3dasm_Lecture10.ipynb) | Chapter 11 | ```HW3 due``` & [HW4 assigned](Assignments/3dasm_Homework4.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture10/3dasm_Lecture10.ipynb) |
| Fri 10/2  | **Linear Regression**: Linear Least Squares model (Gaussian likelihood, Uniform prior, and Point Estimate) | [Lecture 11](Lectures/Lecture11/3dasm_Lecture11.ipynb) | Chapter 11 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture11/3dasm_Lecture11.ipynb) |
| Mon 10/5  | **Linear Regression**: Ridge, Lasso and Bayesian linear regression models (different likelihoods, priors and posteriors) | [Lecture 12](Lectures/Lecture12/3dasm_Lecture12.ipynb) | Chapter 11 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture12/3dasm_Lecture12.ipynb) |
| Wed 10/7  | **Gaussian process regression**: theory | [Lecture 13](Lectures/Lecture13/3dasm_Lecture13.ipynb) | Section 17.2 + Rasmussen's book | ```HW4 due``` & [HW5 assigned](Assignments/3dasm_Homework5.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture13/3dasm_Lecture13.ipynb) |
| Fri 10/9  | **Gaussian process regression**: One dimensional tutorial (Part I: noiseless) | [Lecture 14](Lectures/Lecture14/3dasm_Lecture14.ipynb) | Section 17.2 + Rasmussen's book |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture14/3dasm_Lecture14.ipynb) |
| Mon 10/12  | HOLIDAY 🥹 |  |  |  |  |
| Wed 10/14  | **Gaussian process regression**: One dimensional tutorial (Part II: noisy) | [Lecture 14 (continued)](Lectures/Lecture14/3dasm_Lecture14.ipynb) | Section 17.2 + Rasmussen's book |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture14/3dasm_Lecture14.ipynb) |
| Fri 10/16  | **Gaussian process regression**: Multidimensional tutorial & importance of dataset scaling | [Lecture 15](Lectures/Lecture15/3dasm_Lecture15.ipynb) | Section 17.2 + Rasmussen's book | ```HW5 due``` & [HW6 assigned](Assignments/3dasm_Homework6.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture15/3dasm_Lecture15.ipynb) |
| Mon 10/19  | **Bayesian model selection & Hyperparameter optimization** | [Lecture 16](Lectures/Lecture16/3dasm_Lecture16.ipynb) | Sections 4.6.5 and 17.2.6 + Rasmussen's book Sections 5.2, 5.3 and 5.4 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture16/3dasm_Lecture16.ipynb) |
| Wed 10/21  | Q&A session |  |  | ```HW6 due``` |  |
| Fri 10/23  | **```MIDTERM Exam```** 🦾 |  |  |  |  |
| Mon 10/26  | **Framework for Data-Driven Design & Analysis of Structures & Materials**: [f3dasm](https://f3dasm.readthedocs.io/en/latest/) | [Lecture 17](Lectures/Lecture17/3dasm_Lecture17.ipynb) |  |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture17/3dasm_Lecture17.ipynb) |
| Wed 10/28  | **[f3dasm](https://f3dasm.readthedocs.io/en/latest/) tutorial**: Data-driven process; Sampling methods; Simple model selection example | [Lecture 18](Lectures/Lecture18/3dasm_Lecture18.ipynb) |  | [HW7 assigned](Assignments/3dasm_Homework7.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture18/3dasm_Lecture18.ipynb) |
| Fri 10/30  | **[f3dasm](https://f3dasm.readthedocs.io/en/latest/) tutorial**: General use case (object oriented) | [Lecture 19](Lectures/Lecture19/3dasm_Lecture19.ipynb) |  |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture19/3dasm_Lecture19.ipynb) |
| Mon 11/2  | **```Final Project Overview & Assignment```** 🦾 | [Lecture 20](Lectures/Lecture20/3dasm_Lecture20.ipynb) |  | [Final Project assigned](Projects/3dasm_FinalProject.pdf) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture20/3dasm_Lecture20.ipynb) |
| Wed 11/4  | **Introduction to classification**: Tutorial with 3 simple classifiers on Iris dataset | [Lecture 21](Lectures/Lecture21/3dasm_Lecture21.ipynb) | Sections 1.2.1 + 2.4.2 + 10.1 + 10.2 | ```HW7 due``` & [HW8 assigned](Assignments/3dasm_Homework8.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture21/3dasm_Lecture21.ipynb) |
| Fri 11/6  | **Logistic regression classifier**: Classification as a regression task; Bernoulli observation distribution; the sigmoid trick | [Lecture 22](Lectures/Lecture22/3dasm_Lecture22.ipynb) | Sections 1.2.1 + 2.4.2 + 10.1 + 10.2 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture22/3dasm_Lecture22.ipynb) |
| Mon 11/9  | **Logistic regression classifier**: Point estimate (e.g. MLE) as nonlinear optimization; classification as the mode of PPD | [Lecture 22 (continued)](Lectures/Lecture22/3dasm_Lecture22.ipynb) | Sections 1.2.1 + 2.4.2 + 10.1 + 10.2 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture22/3dasm_Lecture22.ipynb) |
| Wed 11/11  | **Gaussian discriminant analysis (GDA) classifier**: Classification by a *generative* model | [Lecture 23](Lectures/Lecture23/3dasm_Lecture23.ipynb) | Chapter 9 | ```HW8 due``` & [HW9 assigned](Assignments/3dasm_Homework9.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture23/3dasm_Lecture23.ipynb) |
| Fri 11/13  | **Optimization**: Part I | [Lecture 24](Lectures/Lecture24/3dasm_Lecture24.ipynb) | Martins' book Chapters 1 & 4 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture24/3dasm_Lecture24.ipynb) |
| Mon 11/16  | **Optimization**: Part II | [Lecture 25](Lectures/Lecture25/3dasm_Lecture25.ipynb) | Chapters 1 & 4 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture25/3dasm_Lecture25.ipynb) |
| Wed 11/18  | **Optimization**: Part III | [Lecture 26](Lectures/Lecture26/3dasm_Lecture26.ipynb) | Chapters 1 & 4 | ```HW9 due``` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture26/3dasm_Lecture26.ipynb) |
| Fri 11/20  | **Artificial Neural Networks**: Part I | [Lecture 27](Lectures/Lecture27/3dasm_Lecture27.ipynb) | Chapter 13 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture27/3dasm_Lecture27.ipynb) |
| Mon 11/23  | **Artificial Neural Networks**: Part II | [Lecture 28](Lectures/Lecture28/3dasm_Lecture28.ipynb) | Chapter 13 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture28/3dasm_Lecture28.ipynb) |
| Wed 11/25  | **Thanksgiving** 🦃 |  |  |  |  |
| Fri 11/27  | **Thanksgiving** 🦃 |  |  |  |  |
| Mon 11/30  | **Artificial Neural Networks**: Part III | [Lecture 29](Lectures/Lecture29/3dasm_Lecture29.ipynb) | Chapter 13 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture29/3dasm_Lecture29.ipynb) |
| Wed 12/2  | **Artificial Neural Networks**: Part IV | [Lecture 30](Lectures/Lecture30/3dasm_Lecture30.ipynb) |  |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture30/3dasm_Lecture30.ipynb) |
| Fri 12/4  | **Principal Component Analysis** | [Lecture 31](Lectures/Lecture31/3dasm_Lecture31.ipynb) | Section 20.1 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture31/3dasm_Lecture31.ipynb) |
| Mon 12/7  | **Clustering** | [Lecture 32](Lectures/Lecture32/3dasm_Lecture32.ipynb) | Chapter 21 |  | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bessagroup/3dasm_course/blob/main/Lectures/Lecture32/3dasm_Lecture32.ipynb) |
| Wed 12/9  | Q&A session |  |  |  |  |
| Fri 12/11  | **```Final Project presentations```** 🦾 |  |  | ```Final Project report due``` |  |

## Installation instructions

### **OPTION 1**. Run this notebook **locally in your computer**

[Homework 1](Assignments/3dasm_Homework1.pdf) contains detailed instructions to install the virtual environment with all the packages required for this course. Below are more concise instructions for people familiar with mamba and pip:

1. Install Mamba as described [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). (See Homework 1 for additional instructions)

2. Install [git](https://github.com/git-guides/install-git), open command window & clone the repository to your computer:

```
git clone https://github.com/bessagroup/3dasm_course
```

3. Create a virtual environment for this course called '3dasm':

```
mamba create -n 3dasm python=3.11 notebook nb_conda rise numpy scipy matplotlib pandas scikit-learn ipykernel ipywidgets requests
```

4. Activate the '3dasm' virtual environment:

```
mamba activate 3dasm
```

5. Install [pytorch](https://pytorch.org/) in the '3dasm' virtual environment. (See Homework 1 for additional instructions)

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

6. Install f3dasm package:

```
pip install "f3dasm[scipy]==2.4.0"
```

7. Install optuna package:

```
pip install optuna
```

After you installed every package, you are ready to go! Reboot your computer.

* Open a (mamba) command window, activate the '3dasm' environment and start jupyter notebook (it will open in your internet browser):

```
mamba activate 3dasm
jupyter notebook
```

- Open notebook (for example: *3dasm_course/Lectures/Lecture01/3dasm_Lecture1.ipynb*).

You're all set!

### **OPTION 2**. Use **Google's Colab** (no installation required, but Colab session times out if idle)

1. go to [Google Colab](https://colab.research.google.com)
2. Login with your credentials
3. File > Open notebook
4. Click on Github (no need to login or authorize anything)
5. paste the git link: <https://github.com/bessagroup/3dasm_course>
6. click search and then click on the notebook (for example: *3dasm_course/Lectures/Lecture01/3dasm_Lecture1.ipynb*)
