

# Overall structure of an implementation.

An implementation was done in Java-8 it consists of a single class called
HiddenMarkovModel which contains all necessary methods to solve all three problems of HMM.  
For details see: [HMM](http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)

---
# How to run code

First, compile HiddenMarkovModel.java file by the following command from a terminal:

`javac HiddenMarkovModel.java`

Then, use following commands to see outputs for different problems of HMM.

* **Problem 1**

`java HiddenMarkovModel "obsv_prob" `

* **Problem 2**

`java HiddenMarkovModel "viterbi" `

* **Problem 3**

`java HiddenMarkovModel "learn"`

Outputs will be shown on the terminal, if necessary you will be asked for an input.
---

# Detailed explanations

* **Problem 1**

Observation sequence and model parameters are taken as an input from testdata.txt and model.txt files respectively. Then, the probability of the observation sequence given the model is computed. Since it is common to get underflows for long observation sequences, I implemented forward procedure using scaling which is described on page 272 of Rabiners paper.
For the observation sequence: 0 1 1 0 1 0, following outputs are printed:

Probability: 0.001484296712034913  
Log probability(natural logarithm): -6.512814213498095  


* **Problem 2**

Here the model outputs meaningful and optimal state sequence which is inferred using Viterbi algorithm. In order to decrease number of computations I used log trick(see page 273 of Rabiner).
For the observation sequence: 0 1 1 0 1 0, following outputs are printed:

Optimal state sequence is: 0 1 1 1 1 1  
Log probability(natural logarithm): -7.408067403771226  
Number of state transitions needed: 1  

* **Problem 3**

This is the most interesting and challenging part. Given multiple observation sequences provided in the data.txt file the aim is to find model parameters which maximize probability of all observation sequences. To solve this problem the Baum-Welch procedure for multiple observation sequences was implemented. State and state transition probabilities were initialized from uniform distribution. Observation probabilities were initialized randomly.


When you run this procedure you will be asked to input number of states, then training procedure will start and information about the model will be printed on HMMdescription.txt file.

For two states, the following rounded outputs are printed on the description file:

Number of states: 2

Model parameters:
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

Model parameters:
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
These parameters are equivalent.
