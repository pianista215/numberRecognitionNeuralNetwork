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

}