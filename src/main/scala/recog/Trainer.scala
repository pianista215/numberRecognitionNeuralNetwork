package recog

import scala.util.Random

/**
 * In charge of train the Neural Network selecting initial values for theta
 * @author Pianista
 */
case class Trainer(network : NeuralNetwork) {
  
  
  /**
   * Calculate the cost of the current neural network for a training set given with the expected results
   * (As we are on a classifying problem we expect a vector of (1,0,0,0...) for detect a 0
   * (0,1,0,0...) for detect a 1 etc
   * 
   */
  def costFunction(trainingSet: TrainingSet, lambda: Double):Double = {
    val costPerExample = trainingSet map {
      case (input,expected) => {
        val calculated = network.apply(input)
        val calcAndExp = calculated zip expected
        println("CalcAndExp "+calcAndExp)
        val diff = calcAndExp map {
          case(calc,exp) => Math.abs(-exp * Math.log(calc) - (1 - exp)*Math.log(1 - calc) )
        }
        diff.sum
      }
    }
    val noReg = costPerExample.sum/trainingSet.length
    
    val hiddenReg = network.hiddenLayer.foldLeft(0.0)((a,b) => a + b.thetasSumSquare)
    val outputReg = network.outputLayer.foldLeft(0.0)((a,b) => a + b.thetasSumSquare)
    
    noReg + lambda/(2*trainingSet.length) * (hiddenReg + outputReg)
  }
  
  /**
   * From Stanford Coursera Machine Learning Course. Recommendation
   */
  def epsilonInitForHidden: Double = Math.sqrt(6) / Math.sqrt(network.hiddenLayer.length + network.inputLayer)
  
   /**
   * From Stanford Coursera Machine Learning Course. Recommendation
   */
  def epsilonInitForOutput: Double = Math.sqrt(6) / Math.sqrt(network.hiddenLayer.length + network.outputLayer.length)
  
  /**
   * Generate a random theta for a neuron in the Hidden layer
   */
  def initThetaForHidden: List[Double] = 
    List.fill(network.inputLayer+1)(-epsilonInitForHidden + (2*epsilonInitForHidden)* Random.nextDouble())
 
  /**
   * Generate a random theta for a neuron in the Output layer
   */
  def initThetaForOutput : List[Double] = 
    List.fill(network.hiddenLayer.length+1)(-epsilonInitForOutput + (2*epsilonInitForOutput)* Random.nextDouble())
    
  /**
   * Return the trained neural network
   */
  def train(trainingSet: TrainingSet, lambda: Double): NeuralNetwork  = {
    //Start with random thetas
    val hiddenLayer = (1 to network.hiddenLayer.length) map { x=> Neuron(initThetaForHidden) } toList
    val outputLayer = (1 to network.outputLayer.length) map { x=> Neuron(initThetaForOutput) } toList

    val initialNetwork = NeuralNetwork(network.inputLayer, hiddenLayer, outputLayer)
    
    def trainHelper(trainingSet: TrainingSet, currNetwork: NeuralNetwork): NeuralNetwork = {
      if(trainingSet.isEmpty) currNetwork
      else {
        val example = trainingSet.head
        //TODO: Modify theta
        val newNetwork = NeuralNetwork(currNetwork.inputLayer, currNetwork.hiddenLayer, currNetwork.outputLayer)
        trainHelper(trainingSet.tail, newNetwork)
      }
    }
    
    trainHelper(trainingSet, initialNetwork)
  }
  
  
    
  /**
   * Calculate dessuf3 of the current training example
   * Dessuf3 should be a List with the size of the output layer
   * Each neuron of the layer should be mapped to one dessuf3 (dessuf3_k)
   */
  def dessuf3(trainingExample: Input, expectedResult: Result): List[Double] = {
    //d3 = resultFromTheNeuronNetwork - expected
    val result = network.apply(trainingExample)
    val zipped = result zip expectedResult
    val dessuf3 = zipped map { case (res,exp) => res - exp }
    assert(dessuf3.length == network.outputLayer.length)
    dessuf3
  }
  
  /**
   * Calculate dessuf2 of the current training example
   * Dessuf3 should be a vector with the size of the hidden layer
   * Each neuron of the layer should be mapped to one dessuf2 (dessuf2_k)
   */
  def dessuf2(trainingExample: Input, expectedResult: Result): List[Double] = {
    
    //Theta(Ouputlayer) * d3 .* g'(z2)
    
    val d3 = dessuf3(trainingExample,expectedResult)
    val z2 = network.resultsHiddenLayer(trainingExample)
    val g_pri_z2 = z2 map {x=> sigmoidGradient(x)}
    val matrix = network.outputLayer map {neuron => neuron.theta} 
    val inverted = matrix.transpose
    val preProduct = inverted map {row => row zip d3}
    val product = preProduct map {row =>  row map { case (a,b) => a*b}}
    val resultProduct = product map { row => row.sum}
    
    val dessuf2Pre = resultProduct zip g_pri_z2
    val dessuf2 = dessuf2Pre map { case (a,b) => a*b}
    
    assert(dessuf2.length == network.hiddenLayer.length)
    
    dessuf2
  }
    
}