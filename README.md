

# Overall structure of an implementation.

An implementation was done in Java-8-Oracle it consists of a single class called
HiddenMarkovModel which contains all necessary methods to solve all three problems of HMM.  
For details see: [HMM](http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf) 

---
# How to run code

First we need to go to a folder containing source code and all necessary files(model.txt, testdata.txt, data.txt) to compile HiddenMarkovModel.java file using by the following command from a terminal:

`javac HiddenMarkovModel.java` 

After compiling use the following commands to see outputs for different problems.

* **Problem 1**

`java HiddenMarkovModel "obsv_prob" `

* **Problem 2**

`java HiddenMarkovModel "viterbi" `

* **Problem 3**

`java HiddenMarkovModel "learn"`

Outputs will be provided in the terminal, if necessary you will be asked for an input. See corresponding parts of the report for more details.

---

# Detailed explanations

* **Problem 1**

This part takes observation sequence and model parameters as input from testdata.txt and model.txt files, and then computes the probability of the observation sequence given the model, since for long observation sequence it is common to get underflows, I have implemented forward procedure using scaling which is described in page 272 of Rabiners paper. 
For an observation sequence: 0 1 1 0 1 0 ,  the following outputs are printed:

Probability: 0.001484296712034913  
Log probability(natural logarithm): -6.512814213498095  


* **Problem 2**

This part takes observation sequence and model parameters as input from testdata.txt and model.txt files, and outputs meaningful and optimal state sequence. For this part Viterbi algorithm was implemented, in order to have less computations I have used log trick which is described in 
page 273 of Rabiners paper. 
For an observation sequence: 0 1 1 0 1 0 ,  the following outputs are printed:

Optimal state sequence is : 0 1 1 1 1 1  
Log probability(natural logarithm): -7.408067403771226  
Number of state transitions needed: 1  

* **Problem 3**

This is the most interesting and challenging part. Given multiple observation sequences of data provided in a data.txt file the aim is to adjust model parameters in order to maximize the probability of all observation sequences given the model. In order to accomplish an aim the Baum-Welch procedure for multiple observation sequences was implemented. Initially the model parameters were initialized according to suggestions of Rabiners paper page 273-274 namely  initial state and state transition probabilities were given uniform probabilities. An observation symbol probabilities were initialized randomly. Then the following re-estimation procedure with some optimizations was used in train(dataSequences) method:  
When you run this procedure you will be asked to input number of states from the terminal, by following message,  

*Please enter number of states:* 

After your input the training will start, and number of iterations with loss function which is negative  log likelihood of all observations will be printed. The training will stop after some convergence(i.e. loss functions doesn't change much). I have defined threshold = 0.0000001 and used the following conditipn to test convergence Math.abs(newLoss - oldLoss) > threshold.
So when difference between two consecutive loss functions becomes smaller than a threshold the convergence occurs and information about the model is printed to a text file called HMMdescription.txt.

For the case of two stated the following rounded outputs are printed to the description file:

Number of states: 2 

Model parameters rounded: 
```
Pi 
0.05 0.95 

A 
0.92 0.08 
0.04 0.96 

B 
0.22 0.78 
0.94 0.06
```
**Or,** 

Number of states: 2 

Model parameters rounded: 
```
Pi 
0.95 0.05 

A 
0.96 0.04 
0.08 0.92 

B 
0.94 0.06 
0.22 0.78
```
Which are equivalent parameters just state names (i.e. state 0 to 1 and 1 to zero) are changed. As you can see they are very close to actual parameters. You can get outputs for other number of states too. 
