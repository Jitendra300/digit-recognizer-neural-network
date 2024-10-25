# Digit Recognizer in Python from scratch
This project aims to develop a neural network from scratch (only using numpy)!
    
# Why do it?
To get a deeper knowledge of neural networks and also cause its FUN!!!
    
# Neural Architecture
First we have one input layer which consists of **64** inputs and then we have a hidden layer of **10** neurons and then at last we have output layer of **10** neurons.

# Libraries Used:
- numpy
- sklearn

# Output:
By keeping the training iterations at **5000** and learning rate (alpha) at **0.01** we saw that our model is nearly **85%** correct at predicting.<br><br>
![Output 1](/images/minimalOutcome.png)
<br><br>
By increasing the training iterations to **20000** we nearly achieve **92%** correction on testing data. <br><br>
![Output 2](/images/currentMaxOutcome.png)
<br>
# More Info:
Am I cheating by including numpy when this project was supposed to be done from scratch? Well...good question! But how can we ignore the beautiful numpy??? Maybe it can be a future project to do it fully from scratch ;) <br><br>
Also we are using sklearn cause we need the dataset which includes both image and label. Each image is of **8x8** pixels. <br><br>
For more info check the code or message me!

# Limitations:
Cause this is a fun project and my only aim is to learn neural networks by playing around, I don't have any intentions to make this neural network perform at its best, and also one limitations I have noticed is that the neural network doesn't get anybetter than ***94%***.<br><br>
This maybe due to the network architecture or cause this project is from scratch or maybe cause the neural network has found a local minima instead of global minima. Maybe all this can be overcome if we use different neural architecture, use different activation function, use more neurons in layers or increase the amount of hidden layers or maybe have bigger dataset.<br><br>
We might overcome this challenges in future projects

# Future Project:
Maybe do it from scratch by using no numpy! and maybe not even sklearn [we can store the images as arrays inside of a file]
