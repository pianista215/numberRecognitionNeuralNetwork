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
    
    
  def dessuf3(trainingExample: Input, expectedResult: Result): List[Double] = {
    val result = network.apply(trainingExample)
    val zipped = result zip expectedResult
    zipped map { case (res,exp) => res - exp }
  }
}