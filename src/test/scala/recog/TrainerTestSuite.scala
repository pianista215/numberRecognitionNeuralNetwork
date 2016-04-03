package recog

import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner
import org.scalatest.FunSuite

class TrainerTestSuite extends FunSuite{
  
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
    val epsilon = Trainer(new NeuralNetwork(400, 25, 10)).epsilonInitForHidden
    assert(epsilon > 0.11 && epsilon < 0.13)
  }
  
  test("Epsilon init for Output for 25-10 with 400 inputs neural network") {   
    val epsilon = Trainer(new NeuralNetwork(400, 25, 10)).epsilonInitForOutput
    assert(epsilon > 0.40 && epsilon < 0.42)
  }
  
  test("Dessuf3 test"){
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
    val trainer = Trainer(network)
    
    val output = network.apply(inputTraining1)
    print(trainer.dessuf3(inputTraining1, expectedTraining1))
    assert(trainer.dessuf3(inputTraining1, expectedTraining1) === List(output(0) - 1.0, output(1) - 0.0))
    
    //List(-0.18247560743822533, 0.9525588993699814)
  }
  
  test("Dessuf2 test") {
    
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
    val trainer = Trainer(network)
    
    val d2 = trainer.dessuf2(inputTraining1, expectedTraining1)
    
    val d3 = List(-0.18247560743822533, 0.9525588993699814)
    
    val d3_x_theta2 = List(0.0, d3(0)+d3(1), 0.5*d3(0) + 2.0*d3(1))
    val a2 = List(1.0) ::: (network.resultsHiddenLayer(inputTraining1) map {x => sigmoidGradient(x)})
    
    assert(d2 === List(d3_x_theta2(0)*a2(0), d3_x_theta2(1)*a2(1), d3_x_theta2(2)*a2(2)))
    
  }
  
  
}