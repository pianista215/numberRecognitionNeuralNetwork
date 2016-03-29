package recog

/**
 * For us a Neural network is made by two layers. The hidden layer, and the output layer. 
 * Depending of your needs you could consider add more layers (or remove them)
 * 
 * Each of the neurons of the hidden layer receive all the inputs and generate a response
 * Each of the neurons of the output layer receive all the responses of the hidden layer and generate a response
 * 
 * 
 * In our particular case, in the output layer we have 10 neurons. We will identify our number,
 * looking for the value more close to 1 (the values should be between 0 and 1)
 * In other cases if you don't want to classify and you do a regression 
 * (for example, an estimation of a price, a number that predicts something,
 *  you should use just one neuron in the output layer)
 *  
 * (But you can use more layers/change the layer sizes if the problem is not well fit)
 *  
 * @author Pianista
 */
case class NeuralNetwork(
    inputLayer: Int, //Number of inputs of the neural Network (also called input neurons /input units)
    hiddenLayer: List[Neuron],
    outputLayer: List[Neuron]) {
  
  require(inputLayer > 0)
  require(hiddenLayer.foldLeft(true)((actual,neuron)=> actual && neuron.theta.length == inputLayer + 1))
  require(outputLayer.foldLeft(true)((actual,neuron)=> actual && neuron.theta.length == hiddenLayer.length + 1))
  
  /**
   * a3 = Sigmoid(z3)
   */
  def apply(input: Input): Result = {
    sigmoidList(resultsOutputLayer(input))
  }
  
  /**
   * z2
   */
  def resultsHiddenLayer(input:Input):Result = {
    val inputWithBias = 1.0 :: input
    hiddenLayer map { x => x.apply(inputWithBias) }
  }
  
  /**
   * a2 = Sigmoid(z2)
   */
  def resultsHiddenLayerSigmoided(input:Input): Result = sigmoidList(resultsHiddenLayer(input))
  
  /**
   * z3
   */
  def resultsOutputLayer(input:Input): Result = {
    val resultHiddenWithBias = 1.0 :: resultsHiddenLayerSigmoided(input)
    outputLayer map { x => x.apply(resultHiddenWithBias)}
  }
  
  def sigmoidList(input: List[Double]): List[Double] = input map { x => sigmoid(x)}
  
}