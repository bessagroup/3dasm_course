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
    2. A pdf "3dasm_LectureX slides.pdf" with the slides of the course.
    4. A "your_data" folder that you can use to create data or other things in your own computer.
* The preferred method to follow the course is to look directly into the jupyter notebook, as it contains additional notes and working code.

**Grading**

Homeworks 30%, Midterm 30%, and Final Project 40%.

Homeworks will be graded only with 5 levels: A+ (100%; fully correct), A (90%; has minor error), B (75%; has significant error), C (60%; mostly incorrect but homework was delivered), D (0%, not delivered). If you deliver something with an honest attempt at solving the homework you are assigned 60% for that homework. **Late Homework can only get up to A (90%)**. The worst Homework is removed.

**Course outline for the first half**

| DATE | SUBJECT | Notebook | PDF | Homework |
| :-- | :-- | :-- | :-- | :-- |
| Wed 9/6  | Introduction & Basics of univariate statistics | [Lecture 1](Lectures/Lecture1/3dasm_Lecture1.ipynb) | [Slides](Lectures/Lecture1/3dasm_Lecture1_slides.pdf) | [<font color='green'>HW1 assigned</font>](Assignments/3dasm_Homework1.pdf) |
| Fri 9/8  | Handling data with Pandas | [Lecture 2](Lectures/Lecture2/3dasm_Lecture2.ipynb) | [Slides](Lectures/Lecture2/3dasm_Lecture2_slides.pdf) |  |
| Mon 9/11  | Introducing joint & conditional distributions; Bayes' rule | [Lecture 3](Lectures/Lecture3/3dasm_Lecture3.ipynb) | [Slides](Lectures/Lecture3/3dasm_Lecture3_slides.pdf) |  |
| Wed 9/13   | Multivariate statistics; visualization of joint & conditional distributions | [Lecture 4](Lectures/Lecture4/3dasm_Lecture4.ipynb) | [Slides](Lectures/Lecture4/3dasm_Lecture4_slides.pdf) | <font color='red'>HW1 due</font> & [<font color='green'>HW2 assigned</font>](Assignments/3dasm_Homework2.pdf) |
| Fri 9/15  | Bayesian inference for one hidden rv: Part I | [Lecture 5](Lectures/Lecture5/3dasm_Lecture5.ipynb) | [Slides](Lectures/Lecture5/3dasm_Lecture5_slides.pdf) |  |
| Mon 9/18  | Bayesian inference for one hidden rv: Part II | [Lecture 6](Lectures/Lecture6/3dasm_Lecture6.ipynb) | [Slides](Lectures/Lecture6/3dasm_Lecture6_slides.pdf) |  |
| Wed 9/20  | Bayesian inference for one hidden rv: Part III | [Lecture 7](Lectures/Lecture7/3dasm_Lecture7.ipynb) |  [Slides](Lectures/Lecture7/3dasm_Lecture7_slides.pdf) | <font color='red'>HW2 due</font> & [<font color='green'>HW3 assigned</font>](Assignments/3dasm_Homework3.pdf) |
| Fri 9/22  | Machine Learning without going Bayesian: Point Estimates | [Lecture 8](Lectures/Lecture8/3dasm_Lecture8.ipynb) | [Slides](Lectures/Lecture8/3dasm_Lecture8_slides.pdf) |  |
| Mon 9/25  | Linear Regression: Part I | [Lecture 9](Lectures/Lecture9/3dasm_Lecture9.ipynb) | [Slides](Lectures/Lecture9/3dasm_Lecture9_slides.pdf) |  |
| Wed 9/27  | Linear Regression: Part II | [Lecture 10](Lectures/Lecture10/3dasm_Lecture10.ipynb) | [Slides](Lectures/Lecture10/3dasm_Lecture10_slides.pdf) | <font color='red'>HW3 due</font> & [<font color='green'>HW4 assigned</font>](Assignments/3dasm_Homework4.pdf) |
| Fri 9/29  | Linear Regression: Part III | [Lecture 11](Lectures/Lecture11/3dasm_Lecture11.ipynb) | [Slides](Lectures/Lecture11/3dasm_Lecture11_slides.pdf) |  |
| Mon 10/2  | Linear Regression: Part IV | [Lecture 12](Lectures/Lecture12/3dasm_Lecture12.ipynb)  | [Slides](Lectures/Lecture12/3dasm_Lecture12_slides.pdf) |  |
| Wed 10/4  | Gaussian process regression: Part I | [Lecture 13](Lectures/Lecture13/3dasm_Lecture13.ipynb) | [Slides](Lectures/Lecture13/3dasm_Lecture13_slides.pdf) | <font color='red'>HW4 due</font> & [<font color='green'>HW5 assigned</font>](Assignments/3dasm_Homework5.pdf) |
| Fri 10/6  | Gaussian process regression: Part II | [Lecture 14](Lectures/Lecture14/3dasm_Lecture14.ipynb) | [Slides](Lectures/Lecture14/3dasm_Lecture14_slides.pdf) |  |
| Mon 10/9  | <font color='green'>HOLIDAY</font> 🥹 |  |  |  |
| Wed 10/11  | Gaussian process regression: Part III | [Lecture 15](Lectures/Lecture15/3dasm_Lecture15.ipynb) | [Slides](Lectures/Lecture15/3dasm_Lecture15_slides.pdf) | <font color='red'>HW5 due</font> & [<font color='green'>HW6 assigned</font>](Assignments/3dasm_Homework6.pdf) |
| Fri 10/13  | Bayesian model selection |  [Lecture 16](Lectures/Lecture16/3dasm_Lecture16.ipynb) | Slides [Slides](Lectures/Lecture16/3dasm_Lecture16_slides.pdf) |  |
| Mon 10/16  | Q&A session |  |  |  |
| Wed 10/18  | Q&A session |  |  | <font color='red'>HW6 due</font> |
| Fri 10/20  | No lecture  |  |  |  |
| Mon 10/23  | <font color='red'>Midterm exam</font> 🦾 |  |  |  |
| Wed 10/25  | `f3dasm`: Framework for Data-driven Design and Analysis of Structures and Materials |  [Lecture f3dasm](Lectures/Lecture_f3dasm/presentation.ipynb) |  |  |
| Fri 10/28  | Project 1: Learning to optimize |  [Lecture L2O](Lectures/Lecture_L2O_Project/presentation_l2o_project.ipynb) |  |  |
| Mon 10/30  | Project 2: Supercompressible | [Lecture Supercompressible](Lectures/Lecture_Supercompressible_Project/presentation.ipynb) |  |  |
