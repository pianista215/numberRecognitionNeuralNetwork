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
    assert(neuron2.apply(List(1.0, 8.0)) === 4.0)
    
    val network = NeuralNetwork(List(neuron1), List(neuron2))
    assert(network(input) == List(4.0))
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
    
    assert(neuron21.apply(1.0::List(8.0,14.0)) === 15.0)
    assert(neuron22.apply(1.0::List(8.0,14.0)) === 36.0)
    
    val network = NeuralNetwork(List(neuron11,neuron12), List(neuron21, neuron22))
    assert(network.apply(input) === List(15.0, 36.0))
    
  }
  
}