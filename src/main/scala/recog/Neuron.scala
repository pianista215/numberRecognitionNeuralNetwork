package recog

/**
 * Define a Neuron of a Neural network
 * @author Pianista
 */
case class Neuron(theta: List[Double]) {

  /**
   * Theta must be of the same length as the input.
   * We are not taking care about the sizes because is just an example
   * but a Neuron is finally a function that goes from X(input) to => Double
   * X * theta => Double
   */
  def apply(input:List[Double]) : Double = {
    val zipped = input zip (theta)
    val multiplied = zipped map {case (x,y) => x*y}
    multiplied.sum 
  }
  
}