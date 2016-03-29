package object recog {

  /**
   * An alias for the `Nothing` type.
   *  Denotes that the type should be filled in.
   */
  type ??? = Nothing

  /**
   * An alias for the `Any` type.
   *  Denotes that the type should be filled in.
   */
  type *** = Any

  type TrainingSet = List[(Input, Result)]
  type Input = List[Double]
  type Result = List[Double]
  
 /**
   * Gradient of the sigmoid (used for calculate gradient)
   */
  def sigmoidGradient(x: Double): Double =  sigmoid(x)*(1-sigmoid(x))
  
  def sigmoid(x:Double): Double = 1 / (1 + Math.exp(-x))

}