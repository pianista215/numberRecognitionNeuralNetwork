package recog

object Main extends App {
  
  val theta1 = List(0,1.0,2.0,3.0)
  val theta2 = List(0,2.0)
  
  val input = List(8.0,9.0,10.0)
  
  val network = NeuralNetwork(input.length,List(Neuron(theta1)), List(Neuron(theta2)))
  println("Final  "+network.apply(input))
  
}