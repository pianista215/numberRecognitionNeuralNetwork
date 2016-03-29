package recog

import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner
import org.scalatest.FunSuite

class NeuralNetworkTestSuite extends FunSuite{
  
  test("Basic Neuron") {
		
    val theta = List(0, 2.0, 2.0)
    val neuron = Neuron(theta)
    
    val input = List(1.0, 1.0, 2.0)
    assert(neuron.apply(input)===6.0)
    
	}
  
  test("Basic Neuron 2 ") {
		
    val theta = List(0, 3.0, 3.0)
    val neuron = Neuron(theta)
    
    val input = List(1.0, 3.0, 3.0)
    assert(neuron.apply(input)===18.0)
    
	}
  
  test("Neural Network (1 - 1) with 2 inputs") {
    val theta1 = List(0.0, 1.0, 1.0)
    val theta2 = List(0.0, 0.5)
    val input = List(3.0, 5.0)
    
    val neuron1 = Neuron(theta1)
    assert(neuron1.apply(1.0::input) === 8.0)
    
    val neuron2 = Neuron(theta2)
    assert(neuron2.apply(List(1.0, sigmoid(8.0))) == sigmoid(8.0)*0.5)
    
    val network = NeuralNetwork(input.length, List(neuron1), List(neuron2))
    assert(network(input) == List(sigmoid(sigmoid(8.0)*0.5)))
  }
  
  test("Neural Network (2 - 2) with 2 inputs") {
    val theta11 = List(0.0, 1.0, 1.0)
    val theta12 = List(0.0, 1.0, 2.0)
    
    val input = List(2.0, 6.0)
    
    val neuron11 = Neuron(theta11)
    val neuron12 = Neuron(theta12)
    
    val theta21 = List(0.0, 1.0, 0.5)
    val theta22 = List(0.0, 1.0, 2.0)
    val neuron21 = Neuron(theta21)
    val neuron22 = Neuron(theta22)
    
    assert(neuron11.apply(1.0::input) === 8.0)
    assert(neuron12.apply(1.0::input) === 14.0)
    
    assert(neuron21.apply(1.0::List(sigmoid(8.0),sigmoid(14.0))) === sigmoid(8.0)+sigmoid(14.0)*0.5)
    assert(neuron22.apply(1.0::List(sigmoid(8.0),sigmoid(14.0))) === sigmoid(8.0)+sigmoid(14.0)*2.0)
    
    val network = NeuralNetwork(input.length, List(neuron11,neuron12), List(neuron21, neuron22))
    assert(network.apply(input) === List(sigmoid(sigmoid(8.0)+sigmoid(14.0)*0.5), sigmoid(sigmoid(8.0)+sigmoid(14.0)*2.0)))
    
  }
  
  //TODO: Should add more tests for the cost function
  test("Cost function basic") {
    val theta11 = List(0.0, 1.0, 1.0)
    val theta12 = List(0.0, 1.0, 2.0)
    val neuron11 = Neuron(theta11)
    val neuron12 = Neuron(theta12)
    val theta21 = List(0.0, 1.0, 0.5)
    val theta22 = List(0.0, 1.0, 2.0)
    val neuron21 = Neuron(theta21)
    val neuron22 = Neuron(theta22)
    val lambda = 0
    val network = NeuralNetwork(2, List(neuron11,neuron12), List(neuron21, neuron22))
    
    val inputTraining1 = List(2.0, 6.0)
    val expectedTraining1 = List(1.0, 0.0)
    
    val trainingSet = List( (inputTraining1,expectedTraining1) )
    val trainer = Trainer(network)
    
    assert(trainer.costFunction(trainingSet, lambda) > 3.245 && trainer.costFunction(trainingSet, lambda) < 3.250)
  }
  
  test("Sigmoid gradient") {
    assert(sigmoidGradient(0) === 0.25)
  }
  
  test("Epsilon init for Hidden for 25-10 with 400 inputs neural network") {
    
    val hiddenLayer = List.fill(25)(Neuron(List.fill(401)(0.0)))
    val outputLayer = List.fill(10)(Neuron(List.fill(26)(0.0)))
    val epsilon = Trainer(NeuralNetwork(400, hiddenLayer, outputLayer)).epsilonInitForHidden
    assert(epsilon > 0.11 && epsilon < 0.13)
  }
  
  test("Epsilon init for Output for 25-10 with 400 inputs neural network") {
    
    val hiddenLayer = List.fill(25)(Neuron(List.fill(401)(0.0)))
    val outputLayer = List.fill(10)(Neuron(List.fill(26)(0.0)))
    val epsilon = Trainer(NeuralNetwork(400, hiddenLayer, outputLayer)).epsilonInitForOutput
    assert(epsilon > 0.40 && epsilon < 0.42)
  }
  
}