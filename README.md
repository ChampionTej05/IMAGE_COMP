# Project Title 
### IMAGE-COMPRESSION using CAE

## Getting Started 
This the basic approach of using the CAE to compress the image and recreate them again. 
We have used __Python 3.6.5 :: Anaconda, Inc.__ to make the project. You can get it from here [Anaconda](https://www.anaconda.com/download/)

## Prerequisites:
 1. Basic Python understanding
 2. Knowledge about the Machine learning algorithms 
 3. Functioning of Convolutional Neural Networks


## Running the Script:
```
1. Install the necessary modules (Provided Below)
2. Go to "training_model.py" file and decrease the count=2000 in epoch section to 500/1000, cause your computer might not be able to handle such high processing.
3. Warning: Don't use Virutal box, minimum RAM=6GB is needed to such neural network.
4. Now run the "training_model.py" file and wait till the model gets trained.
5. Open the "Encoding.py" file and run it, input is already feeded to it, just run it in same directory in which i have provided.
6. Open the "Decoding.py" and run it, check for the reconstructed image, plot.
```

## Project Files:
1. Abstract of the Project can be found here [Abstract](https://drive.google.com/open?id=1dkDY-etIBPQBvHE4bc3bJEQDRm7sRX0N)
2. Pipeline of the Project can be found here [Pipeline](https://drive.google.com/open?id=1luwRXRsVp7c2tYVuw5hJnl_vD4Lr9Bga)
3. Software and Algorithms used in the project can be found here [DEV ALGO](https://drive.google.com/open?id=1SQXgXsZGI-Y1wuMIBiYpOT91-8TFxE_k)

## Outcomes:
1. We were successfully able to produce the reconstructed image, with loss in range of 100 to 120.
2. The standalone scripts to encode as well as decode your 28x28 images.
3. The IEEE paper on image compression using CAE [IMAGE_COMP](https://drive.google.com/open?id=1ByZY7KIpHVee0mZhU9cgz0zd0eXuxt3-)

### Results Obtained After using the Optimizer and Before using Optimizer
![ADAM Optimizer](https://github.com/ChampionTej05/IMAGE_COMP/blob/master/images/all_four_plots.png)

### Outcome for test image of 4. Image was imported from MNIST data set
![Image Outcomes](https://github.com/ChampionTej05/IMAGE_COMP/blob/master/images/four_result.png)

### Experimental Analysis of the loss, when batch size of 16 and 8 were tried
![Batch Wise Outcome](https://github.com/ChampionTej05/IMAGE_COMP/blob/master/images/only_two_plots.png)

## Limitations: 
1. Our model currently accepts only 28x28 images, so your image would be resized to 28x28 if it is greater than that.
2. Our model is currently trained on only MNIST data set, so it might not perform as it was expected on real world images.
3. The average loss over the period of 2000 is below 100, but we are yet to reach point of saturation.
4. This project is the basic implemenation of Neural Network conceptualization and hence we have not yet considered the techniques like PCA , DenseNET and GAN to create better complex architecture.

## Future Goals:
1. Reduce the average loss to below 50
2. Make it available for all types of image sizes
3. Use of denseNET to achieve the lossless image compression.
4. Training model over real world dataset of low resolution images

## Modules used:
1. Tensorflow version  1.12.0 
2. Numpy version 1.14.3
3. Open cv version 3.4.1
4. Matplotlib version 2.2.2

## Acknowledgments: 
1. Huge vote of thanks to ExpertsHub for providing us the knowledge to explore field of Machine learning.
2. Research paper from [Research gate ] (https://www.researchgate.net) really helped us to drive the project continiously. 
3. Great thanks to our Mentor Nimish Sir and Shubham Sir for helping us in project.

