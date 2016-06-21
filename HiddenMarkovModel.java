import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class HiddenMarkovModel {
	
	double[][]   a          ;
	double[][]   b          ;
	double[]     pi         ;
	double[]     c          ;
	int          numOfStates;
	int          numOfObs   ;
	double       stateSeqPrb;
	
	/**
	 * Constructor for HMM class
	 * @param a            - state transition matrix
	 * @param b            - observation symbol probability matrix
	 * @param pi           - initial state distribution
	 * @param numOfStates  - # of states in the model
	 * @param numOfObs     - # of distinct observation symbols
	 */
	public HiddenMarkovModel(double[][] a, double[][] b, double[] pi, int numOfStates, int numOfObs)
	{
		this.a           = a          ;
		this.b           = b          ;
		this.pi          = pi         ;
		this.numOfStates = numOfStates;
		this.numOfObs    = numOfObs   ;
	}
	
	/**
	 * Implementation of forward procedure, with scaling in order to avoid overflow/underflow
	 * For scaling look at: R.Rabiner page 272
	 * @param observations - observation sequence
	 * @return alpha       - probability of partial observation sequence
	 */
	public double[][] forward(ArrayList<Integer> observations)
	{
		int T            = observations.size()       ;
		double[][] alpha = new double[T][numOfStates];
		c                = new double[T]             ;
		double tempSum   = 0;
		
		//Initialization step
		for(int i = 0;i < numOfStates; i++)
		{
			alpha[0][i] = pi[i] * b[i][observations.get(0)];
			tempSum    += alpha[0][i];
		}
		
		//Scaling
		c[0] = 1 / tempSum;
		for(int i = 0; i < numOfStates; i++)
		{
			alpha[0][i] = alpha[0][i] * c[0];
		}
		
		//Induction step
		double temp;
		
		for( int t = 0; t < T - 1; t++ )
		{
			tempSum = 0;
			for(int j = 0; j < numOfStates; j++)
			{
				temp = 0;
				for( int i = 0; i < numOfStates; i++ )
				{
					temp += alpha[t][i]*a[i][j];
				}
				alpha[t+1][j] = temp * b[j][observations.get(t+1)];
				tempSum      +=  alpha[t+1][j];
				
			}
			
			//Scaling
			c[t+1] = 1 / tempSum;
			for(int k = 0; k < numOfStates; k++)
			{
				alpha[t+1][k] = c[t+1] * alpha[t+1][k];
			}
		}
		
		return alpha;
		
	}
	
	/**
	 * Implementation of backward procedure, with scaling in order to avoid overflow/underflow
	 * For scaling look at: R.Rabiner page 272
	 * @param observations - observation sequence
	 * @return beta        - probability of partial observation sequence
	 */
	public double[][] backward( ArrayList<Integer> observations )
	{
		int T           = observations.size()       ;
		double[][] beta = new double[T][numOfStates];
		
		//Initialization
		for(int i = 0; i < numOfStates; i++)
		{
			beta[T-1][i] = 1;
		}
		
		//Scaling
		for(int i = 0; i < numOfStates; i++)
		{
			beta[T-1][i] = c[T-1] * beta[T-1][i];
		}
		
		
		//Induction
		for( int t = T-2; t >= 0; t-- )
		{
			for(int i = 0; i < numOfStates; i++)
			{
				for(int j = 0; j < numOfStates; j++)
				{
					beta[t][i] += a[i][j] * b[j][observations.get(t + 1)] * beta[t+1][j];
				}
			}
			
			//Scaling
			for(int k = 0; k < numOfStates; k++)
			{
				beta[t][k] = beta[t][k] * c[t];
			}
		}
		
		
		return beta;
	}
	
	/**
	 * This is helper method in order to visualize various matrices
	 * This method was created for debugging
	 * @param matrix - any 2D matrix(beta, alpha, gamma)
	 */
	public void showMatrix(double[][] matrix)
	{
		//Print contents of the matrix
		for(int i = 0; i < matrix.length; i++)
		{
			for(int j = 0; j < matrix[0].length; j++)
			{
				System.out.print( matrix[i][j] + " " );
			}
			System.out.println();
		}
	}
	
	/**
	 * Viterbi algorithm which finds single best state sequence for a given observation sequence
	 * Implemented using logarithms to avoid numerical problems and have significantly less computation
	 * For reference look at: R.Rabiner page 273
	 * @param observations - observation sequence
	 * @return stateSeq    - best state sequence for given observation sequence
	 */
	public int[] viterbi(ArrayList<Integer> observations)
	{
		int T            = observations.size()       ;
		double[][] sigma = new double[T][numOfStates];
		int[][] psi      = new int[T][numOfStates]   ;
		
		//Initialization
		for(int i = 0; i < numOfStates; i++)
		{
			sigma[0][i] = Math.log(pi[i]) + Math.log(b[i][observations.get(0)]) ;
			psi[0][i]   = 0;
		}
		
		//Recursion step
		double max   = -Double.MAX_VALUE;
		int maxIndex = 0;
		for(int t = 1; t < T; t++)
		{
			for(int j = 0; j < numOfStates; j++)
			{
				max = -Double.MAX_VALUE;
				for(int i = 0; i < numOfStates; i++ )
				{
					if( max < sigma[t-1][i] + Math.log( a[i][j] )  )
					{
						max      = sigma[t-1][i] + Math.log( a[i][j] ) ;
						maxIndex = i;
					}
				}
				sigma[t][j] = max + Math.log( b[j][observations.get(t)]);
				psi[t][j]   = maxIndex;
			}
		}
		
		//Termination
		double logpStar = -Double.MAX_VALUE;
		int qStar       = 0;
		for(int i = 0; i < numOfStates; i++)
		{
			if(logpStar < sigma[T-1][i])
			{
				logpStar = sigma[T-1][i];
				qStar    = i;
			}
		}
		stateSeqPrb = logpStar;
		
		//Find path by backtracking
		int[] stateSeq = new int[T];
		stateSeq[T-1]  = qStar;
		
		for( int t = T - 2; t >= 0; t-- )
		{
			stateSeq[t] = psi[ t+1 ][ stateSeq[ t+1 ] ];
		}
		
		return stateSeq;
	}
	
	/**
	 * This method gets test sequence of observations
	 * @param fileName - name of file with test sequence( observations )
	 * @return obs     - observations from file
	 */
	public ArrayList<Integer> getTestData(String fileName)
	{
		//ArrayList of observations
		ArrayList<Integer> obs = new ArrayList<Integer>();
		
		try {
			//Open file and read all observations
			Scanner sc = new Scanner( new File(fileName) );
			while(sc.hasNext())
			{
				obs.add(sc.nextInt());
			}
			sc.close();
			
		}
		catch (FileNotFoundException e) 
		{
			System.out.println("File wasn't found");
			e.printStackTrace();
		}
		
		return obs;
	}
	
	/**
	 * This method initializes parameters of HMM with inputs from text file
	 *  and returns instance HiddenMarkovModel class
	 * @param fileName    - name of file with HMM parameters
	 * @param numOfStates - number of states
	 * @param numOfObs    - number of observable's
	 * @return HMM        - instance of HiddenMarkovModel class
	 */
	public static HiddenMarkovModel initializeParameters(String fileName, int numOfStates, int numOfObs)
	{
		String tmp                                         ;
		double[][] a = new double[numOfStates][numOfStates];
		double[][] b = new double[numOfStates][numOfObs]      ;
		double[]  pi = new double[numOfStates]             ;
		
		try {
			//Open file with HMM parameters
			Scanner sc = new Scanner( new File(fileName));
			while(sc.hasNext())
			{
				tmp = sc.next();
				if(tmp.equals("A"))
				{
					//Initialize a - state transition matrix
					sc.nextLine();
					for(int i = 0; i < numOfStates; i++)
					{
						for(int j = 0; j < numOfStates; j++)
						{
							a[i][j] = sc.nextDouble();
						}
					}
				}
				else if(tmp.equals("B"))
				{
					//Initialize b - observation symbol probability matrix
					sc.nextLine();
					for(int i = 0; i < numOfStates; i++)
					{
						for(int j = 0; j < numOfObs; j++)
						{
							b[i][j] = sc.nextDouble();
						}
					}
				}
				else if(tmp.equals("Pi"))
				{
					//Initialize pi - initial state probabilities
					sc.nextLine();
					for(int i = 0; i < numOfStates; i++)
					{
						pi[i] = sc.nextDouble();
					}
				}
			}
			
			//close scanner
			sc.close();
			
		} 
		catch (FileNotFoundException e1) 
		{
			System.out.println("File wasn't found");
			e1.printStackTrace();
		}
		
		
		return new HiddenMarkovModel(a, b, pi, numOfStates, numOfObs);
	}
	
	/**
	 * This method initializes parameters of HMM randomly and uniformly
	 * and returns instance of HiddenMarkovModel class
	 * For details of initialization step look at: R.Rabiner page 273
	 * @param numOfStates - number of states
	 * @param numOfObs    - number of observable's 
	 * @return HMM        - instance of HiddenMarkovModel class
	 */
	public static HiddenMarkovModel initializeParametersRandomly(int numOfStates, int numOfObs)
	{
		double[][] a = new double[numOfStates][numOfStates];
		double[][] b = new double[numOfStates][numOfObs]      ;
		double[]  pi = new double[numOfStates]             ;
		Random rand  = new Random()                        ;
		
		
		//Uniformly initialize pi
		for(int i = 0; i < numOfStates; i++)
		{
			pi[i] = 1.0 / numOfStates;
		}
		
		//Uniformly initialize a
		for(int i = 0; i < numOfStates; i++)
		{
			for(int j = 0; j < numOfStates; j++)
			{
				a[i][j] = 1.0/ numOfStates;
			}
		}
		
		//For case of 2 observable's initialize b randomly
		double rnum;
		
		for(int i = 0; i < numOfStates; i++)
		{
			do
			{
				rnum = rand.nextDouble();
				b[i][0] = rnum;
				b[i][1] = 1.0 - rnum; 
				
			}while(rnum == 0.0 || rnum == 1.0);
		}
		
		
		return new HiddenMarkovModel(a, b, pi, numOfStates, numOfObs);
	}
	
	/**
	 * This method gets multiple observations sequences for training purposes
	 * @param fileName - name of file with multiple observations sequences
	 * @return data    - ArrayList of ArrayLists containing all observations
	 */
	public ArrayList<ArrayList<Integer>> getTrainingData(String fileName)
	{
		//ArrayList of ArrayLists to store multiple observation sequences
		ArrayList< ArrayList<Integer> >  data = new ArrayList< ArrayList<Integer> >();
		
		try {
			//Open file
			Scanner sc = new Scanner( new File(fileName));
			while(sc.hasNextLine())
			{
				//Read each observation sequence into one ArrayList
				ArrayList<Integer> oneSeq = new ArrayList<Integer>();
				String line = sc.nextLine();
				for(int i = 0; i < line.length(); i++)
				{
					if(line.charAt(i) !=  ' ')
					{
						oneSeq.add((int) line.charAt(i) - '0');
					}
				}
				//Add observation sequence to the data
				data.add(oneSeq);
			}
			sc.close();
			
		} 
		catch (FileNotFoundException e) 
		{
			System.out.println("File wasn't found");
			e.printStackTrace();
		}
		return data;
		
	}
	
	/**
	 * This method implements Baum-Welch algorithm for multiple sequences of training data
	 * For reference look at: R.Rabiner page 273
	 * @param trainingData - all training sequences as ArrayList of ArrayLists
	 * @return loss        - loss function(log likelihood of whole observation sequence)
	 */
	public double train(ArrayList<ArrayList<Integer>>  trainingData)
	{
		//Initialize variables needed for algorithm
		double[][] alpha;
		double[][] beta ;
		double     prob ;
		
		//Numerator and denominator for A (state transition matrix)
		double[][] totalNumA   = new double[a.length][a[0].length];
		double[][] totalDenomA = new double[a.length][a[0].length];
		
		//Numerator and denominator for B (observation symbol probability matrix)
		double[][] totalNumB   = new double[b.length][b[0].length];
		double[][] totalDenomB = new double[b.length][b[0].length];
		
		//Numerator and denominator for Pi (initial state probabilities)
		double[] totalNumPi   = new double[pi.length];
		double[] totalDenomPi = new double[pi.length];
		
		//And other variables
		double[][] gamma                             ; 
		double num = 0, denum = 0, temp = 0, loss = 0;
		
		
		
		for( int k = 0; k < trainingData.size(); k++ )
		{
			//Compute alpha, beta and gamma matrices
			alpha = forward(trainingData.get(k));
			beta  = backward(trainingData.get(k));
			gamma = getGamma(alpha, beta);
			//Compute probability of observation sequence
			prob  = getProbability();
			if(prob == 0)
			{
				System.err.println("Invalid zero probability");
				System.err.println(prob);
			}
			
			//Update loss function
			loss += Math.log(prob);
			
			//Accumulate numerator and denominator arrays needed to re-estimate initial state  probabilities
			for(int i = 0; i < pi.length; i++)
			{
				totalNumPi[i]   +=  gamma[0][i];
				totalDenomPi[i] += 1;
			}
			
			//Accumulate numerator and denominator matrices needed to re-estimate A - state transition matrix
			for(int i = 0; i < a.length; i++)
			{
				for(int j = 0; j < a[0].length; j++)
				{
					num   = 0;
					denum = 0;
					for(int t = 0;  t < trainingData.get(k).size() - 1; t++ )
					{
						num   +=  alpha[t][i] * a[i][j] * b[j][trainingData.get(k).get(t + 1)] * beta[t+1][j];
						temp   = 0;
						for(int u = 0; u < numOfStates; u++)
						{
							temp +=  alpha[t][i] * a[i][u] * b[u][trainingData.get(k).get(t + 1)] * beta[t+1][u];
						}
						denum += temp; 
						
					}
					totalNumA[i][j]   +=  num;
					totalDenomA[i][j] += denum;
				}
			}
			
			//Accumulate numerator and denominator matrices needed to re-estimate B - observation symbol probability matrix
			for(int j = 0; j < b.length; j++)
			{
				for(int l = 0; l < b[0].length; l++)
				{
					num   = 0;
					denum = 0;
					for(int t = 0; t < trainingData.get(k).size(); t++)
					{
						if(trainingData.get(k).get(t) == l)
						{
							num += gamma[t][j];
						}
						denum += gamma[t][j];
					}
					
					totalNumB[j][l]   +=  num;
					totalDenomB[j][l] += denum;
				}
			}
		}
	
		//Re-estimate initial state  probabilities
		for(int i = 0; i < pi.length; i++)
		{
			pi[i] = totalNumPi[i] / totalDenomPi[i];
		}
		
		//Re-estimate A - state transition matrix
		for(int i = 0; i < a.length; i++)
		{
			for(int j = 0; j < a[0].length; j++ )
			{
				a[i][j] = totalNumA[i][j]/totalDenomA[i][j];
			}
		}
		
		//Re-estimate B - observation symbol probability matrix
		for(int i = 0; i < b.length; i++)
		{
			for(int j = 0; j < b[0].length; j++ )
			{
				b[i][j] = totalNumB[i][j] / totalDenomB[i][j];
			}
		}
		
		return -loss;	
	}
	
	/**
	 * Compute probability of particular observation sequence
	 * For reference look at: R.Rabiner page 273
	 * @return ans - probability of particular observation sequence
	 */
	public double getProbability()
	{
		double ans = 1;
		for(int i = 0; i < this.c.length; i++)
		{
			ans *= c[i];
		}
		return 1 / ans;
	}
	
	/**
	 * This method computes probability of being in particular state i at time t
	 * given observation sequence and model
	 * For reference look at: R.Rabiner page 263
	 * @param alpha   - forward variable
	 * @param beta    - backward variable
	 * @return gamma  - gamma variable
	 */
	public double[][] getGamma(double[][] alpha, double[][] beta)
	{
		//Initialize gamma matrix
		double[][] gamma = new double[beta.length][beta[0].length];
		double     denum = 0                                      ;
		
		//Compute gamma matrix
		for(int t = 0; t < beta.length; t++ )
		{
			denum = 0;
			for(int i = 0; i < beta[0].length; i++)
			{
				denum += alpha[t][i] * beta[t][i];
			}
			
			for(int i = 0; i < beta[0].length; i++)
			{
				gamma[t][i] = alpha[t][i]*beta[t][i] / denum;
			}
		}
		
		//return gamma matrix
		return gamma;
	}
	
	/**
	 * This method trains HMM model using training data and Baum-Welch algorithm
	 * Training stops when loss function changes insignificantly after each iteration
	 * @param trainingData - all training sequences as ArrayList of ArrayLists
	 */
	public void learn( ArrayList<ArrayList<Integer>>  trainingData )
	{
		//Perform initial training
		double oldLoss = train(trainingData);
		double newLoss = train(trainingData);
		
		//Define a threshold for training, you can change this if training takes infinitely long 
		double threshold = 0.0000001;
		int epoch = 1;
		
		//Train HMM and learn optimal parameters of the model
		System.out.println("Training has started!");
		while( Math.abs(newLoss - oldLoss) > threshold )
		{
			oldLoss = newLoss;
			newLoss = train(trainingData);
			System.out.println("Epoch: " + epoch + " loss: " +newLoss);
			epoch++;
		}
		System.out.println("Training is finished!");
	}
	
	/**
	 * This method prints model parameters to a file HMMdescription.txt
	 */
	public void printModel()
	{
		File file = new File("HMMdescription.txt");
		
		try {
			
			file.createNewFile();
			if(file.exists())
			{
				PrintWriter pw = new PrintWriter(file);
				pw.println("Number of states: " + numOfStates);
				pw.println();
				pw.println("Model parameters rounded:");
				pw.println();
				pw.println("Pi");
				pw.println();
				
				for(int i = 0; i < pi.length; i++)
				{
					pw.printf("%.2f ", pi[i]);
				}
				pw.println();
				pw.println();
				pw.println("A");
				pw.println();
				for(int i = 0; i < a.length; i++)
				{
					for(int j = 0 ; j < a[0].length; j++)
					{
						pw.printf("%.2f ", a[i][j]);
					}
					pw.println();
				}
				
				pw.println();
				pw.println("B");
				pw.println();
				for(int i = 0; i < b.length; i++)
				{
					for(int j = 0 ; j < b[0].length; j++)
					{
						pw.printf("%.2f ", b[i][j]);
					}
					pw.println();
				}
				pw.flush();
				pw.close();
				
			}
		} 
		catch (FileNotFoundException e1) 
		{
			System.out.println("File wasn't found");
			e1.printStackTrace();
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
		
	}
	

	public static void main(String[] args) {
		
		int numOfStates       = 2             ;
		int numOfObs          = 2             ;
		String command        = args[0]       ;
		HiddenMarkovModel hmm = null          ;
		String testFileName   = "testdata.txt";
		String modelFileName  = "model.txt"   ;
		
		//Solution to problem 1
		if(command.equals("obsv_prob"))
		{
			//Initialize parameters from model.txt file and get test observation from testdata.txt
			hmm = initializeParameters(modelFileName, numOfStates, numOfObs);
			ArrayList<Integer> observations = hmm.getTestData(testFileName);
			//Run forward procedure
			hmm.forward(observations);
			double part1Ans = 1, logProb = 0;
			
			for(int i = 0; i < observations.size(); i++)
			{
				part1Ans *=  hmm.c[i];
				logProb  += Math.log(hmm.c[i]);
				
			}
			System.out.println("Probability: " + 1/part1Ans);
			System.out.println("Log probability(natural logarithm): " + ( -logProb ));
		} 
		else if(command.equals("viterbi")) //Solution to problem 2
		{
			//Initialize parameters from model.txt file and get test observation from testdata.txt
			hmm = initializeParameters(modelFileName, numOfStates, numOfObs);
			ArrayList<Integer> observations = hmm.getTestData(testFileName);
			//Run Viterbi algorithm
			int[] stateSeq = hmm.viterbi(observations);
			
			System.out.print("Optimal state sequence is : ");
			int currState;
			int prevState;
			int count = 0;
			System.out.print(stateSeq[0]+ " ");
			for(int i = 1; i < stateSeq.length; i++)
			{
				prevState = stateSeq[i-1];
				currState = stateSeq[i]  ;
				System.out.print(currState + " ");
				if(prevState != currState)
				{
					count++;
				}
			}
			System.out.println();
			System.out.println("Log probability(natural logarithm): " + hmm.stateSeqPrb);
			System.out.println("Number of state transitions needed: " + count);
		}
		else if(command.equals("learn"))  //Solution to problem 3
		{
			//Get number of states as input
			Scanner sc = new Scanner(System.in);
			System.out.print("Please enter number of states: ");
			numOfStates = sc.nextInt();
			sc.close();
			//Initialize parameters of the model randomly
			hmm = initializeParametersRandomly( numOfStates, numOfObs);
			//Get training sequence
			ArrayList<ArrayList<Integer>>  trainingData = hmm.getTrainingData("data.txt");
			//Learn model parameters and print them to a file
			hmm.learn(trainingData);
			hmm.printModel();
		}

	}	
}
