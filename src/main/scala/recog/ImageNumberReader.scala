package recog

import javax.imageio.ImageIO
import java.awt.Color
import java.io.File

/**
 * Reader object for patterns
 * @author Pianista
 */
object ImageNumberReader {
  
  
  /**
   * An image is just a list o pixels (0,1) 0-> white 1-> black (We use double because out neural network is agnostic and used doubles as input)
   */
  type Image = List[Double]
  
  /**
   * The pattern contains all the images of the number (1.1.png, 1.2.png...)
   */
  type Pattern = List[Image]
  
  val NumberOfImagesPerNumber = 9
  
  lazy val patterns : List[(Int,Pattern)] = {
    
    /**
     * Retrieve the list of images of the pattern
     */
    def loadPattern(number: Int): Pattern = {
      (1 to NumberOfImagesPerNumber).toList map { x => loadImage(number,x)}
    }
    
    /**
     * Retrieve the List of pixels of the image
     */
    def loadImage(number: Int, count: Int): Image = {
      val img = ImageIO.read(getClass.getResourceAsStream(number.toString+"."+count.toString+".png"))
      val pixels = img.getRGB(0, 0, img.getWidth, img.getHeight, null, 0, img.getWidth).toList
      pixels map {x => if(new Color(x).getRed() == 255) 0.0 else 1.0} //White 0 Black 1
    }
   
    (0 to 9).toList map {x => (x,loadPattern(x))}
  }
  
  
}