package recog

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
   * Gradient of the sigmoid (used for calculate gradient)
   */
  def sigmoidGradient(x: Double): Double =  sigmoid(x)*(1-sigmoid(x))
  
  def sigmoid(x:Double): Double = 1 / (1 + Math.exp(x))
  
  /**
   * From Stanford Coursera Machine Learning Course. Recommendation
   */
  def epsilonInitForHidden: Double = Math.sqrt(6) / Math.sqrt(network.hiddenLayer.length + network.inputLayer)
  
   /**
   * From Stanford Coursera Machine Learning Course. Recommendation
   */
  def epsilonInitForOutput: Double = Math.sqrt(6) / Math.sqrt(network.hiddenLayer.length + network.outputLayer.length)
  
}