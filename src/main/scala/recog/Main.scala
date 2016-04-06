package recog
/**
 * @author Unai Sarasola
 * 
 * In this program we are going to train a Neural Network in order to identify the numbers in a image PNG of 8x8 pixels
 * After that we will try out trained Network with a test.png image to see how the network is able to identify the number on it
 * 
 * You can play with the lambda, step, and number of iterations changing Trainer() with Trainer(iter, lambda, step) to obtain different networks
 * 
 * If you use that an approach like that into a production environment, you should change my fmincg function, by a better solution from a Common library
 * 
 */
object Main extends App {
  
  
  println("Loading examples...")
  
  val trainingPatterns = ImageNumberReader.patterns
  
  println("Examples loaded...")
  
  println("Creating neural network...")
  
  val hiddenLayerSize = 20
  val outputLayerSize = 10
  val inputs = 64 //8x8 pixels
  
  val network = new NeuralNetwork(inputs, hiddenLayerSize, outputLayerSize)
  
  val trainer = Trainer()
  
  val trainingExamples = trainingPatterns flatMap {
    case (number, images) => images map {
      img => (img, number)
    }
  }
  
  //we have to convert the number to a List of 0's where the position is 1 Ex 1-> (0,1,0,0...) 2-> (0,0,1,0...)
  
  val trainingExamplesFormmated = trainingExamples map { case (img, number) => {
    val empty = List.fill(10)(0.0)
    (img, empty updated(number, 1.0))
  } }
  
  val trainedNetwork = trainer.train(network, trainingExamplesFormmated)
  
  //Once we have the trainedNetwork, test how can fit the trainingSet
  val examplesCorrect = trainingExamplesFormmated map {
    case (input, expectedRes) => {
      val res = trainedNetwork.apply(input)
      if(res.indexOf(res.max) == expectedRes.indexOf(expectedRes.max)) 1.0 else 0.0
    }
  }
  
  val percentage = (examplesCorrect.sum/examplesCorrect.length)*100
  println("% correct: "+percentage)
  
  
  //Try to identify the target image
  val test = ImageNumberReader.loadImage("test.png")
  val result = trainedNetwork.apply(test)
  println("Test image identified as number: "+ result.indexOf(result.max))
  
}