# Digit Recognizer in Python from scratch
This project aims to develop a neural network from scratch [no PyTorch, no TensorFlow!!]
    
# Why do it?
To get a deeper knowledge of neural networks and also cause its FUN!!!
    
# Neural Architecture
First we have one input layer which consists of **64** inputs and then we have a hidden layer of **10** neurons and then at last we have output layer of **10** neurons. We have default epoch size of 65, batchsize of 8, learningrate[alpha] of 0.03 and datasplit of 1000. We can change this default values from command arguments of the file.

# Libraries Used:
- numpy
- sklearn

# Installation:
    git clone https://github.com/Jitendra300/digit-recognizer-neural-network.git
    cd digit-recognizer-neural-network
    pip install -r requirements.txt
    python main.py

# Demo:
    python main.py --epoch=50 --learning_rate=0.01 --batch_size=16 --datasplit=1200
    
# Output:
By keeping the default values namely epoch=65, learning_rate=0.03, batch_size=8, datasplit=1000 we get an accuracy of 92.8481 <br>
![Output 1](/images/output.png)
<br><br>

# More Info:
Am I cheating by including numpy when this project was supposed to be done from scratch? Well...good question! But how can we ignore the beautiful numpy??? Maybe it can be a future project to do it fully from scratch ;) <br><br>
Also we are using sklearn cause we need the dataset which includes both image and label. Each image is of **8x8** pixels. <br><br>
For more info check the code or message me!

# Limitations:
Cause this is a fun project and my only aim is to learn neural networks by playing around, I don't have any intentions to make this neural network perform at its best, and also one limitations I have noticed is that the neural network doesn't get anybetter than ***93%***.<br><br>
This maybe due to the network architecture or cause this project is from scratch or maybe cause the neural network has found a local minima instead of global minima. Maybe all this can be overcome if we use different neural architecture, use different activation function, use more neurons in layers or increase the amount of hidden layers or maybe have bigger dataset.<br><br>
We might overcome this challenges in future projects

# Future Project:
Maybe do it from scratch by using no numpy! and maybe not even sklearn [we can store the images as arrays inside of a file]
