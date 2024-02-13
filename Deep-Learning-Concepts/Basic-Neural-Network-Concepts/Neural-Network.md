# The Big Picture

We can think of a neural network (NN) as a mathematical function which takes an input data x and computes an output y:
y = fNN (x)
For example, a model trained to identify spam emails takes in an email as input data x, and output 0 or 1 indicating whether the email is spam.
The function fNN has a particular form: it’s a nested function. In lecture, we learnt the concepts of network layers. So, for a 3-layer neural network that returns a scaler, fNN looks like this:
y = fNN (x) = f3(f2(f1(x)))
In the above equation, f1 and f2 are vector functions of the following form:
fl(z) = gl(Wl · z + bl)
where l is called the layer index. The function gl is called an activation function (e.g. ReLU, Sigmoid). The parameters Wl (weight matrix) and bl (bias vector) for each layer are learnt using gradient descent by optimizing a particular loss function5 depending on the task.
4.1 Understanding the shapes
1. A0 (inputs): Instead of passing each training data ( input features of dimension Cin) we consider a batch of them at once because we are doing the same computation for each of the input features hence (N × Cin, where N is batch number)
2. W0 (weight matrix): From lectures, we know the value of each neuron is the affine combination of the input, weight and bias (W · A + b for this multiplication to be compatible, the second dimension of W must match the first dimension of A. However this is for a single neuron, for multiple neurons, the first dimension of W should be equal to the number of output neurons Cout hence the shape Cout × Cin
3. B0 (Biases): From the previous explanation about the weights, we can infer that the bias for a single input feature should be 1 × 1, however, since we are considering a batch of N inputs at once, the biases shape become N × 1.
4. Z0: Z0 is the output is the affine combination of input, weights and biases and we require it to be of shape N × Cout so that each of the inputs in the batch have their outputs for the number of neurons in Cout. To see how this works, check example below.
5. f: The activation is a linear function and does not change the shape of the input
6. A1: This is the output of the activation function and hence same shape as Z0
7. Y: After the activation of the hidden layer, the network can be made deeper by adding several layers. However, the output of the final layer should match your desired output shape, in this case, Cout of the final layer is same as the Cout of the weight which is equal to the number of neurons because we have just one layer. Cout will be the equal to number of neurons in your final layer.

# Neural Networks Overview

In this overview, we explore the functionality of neural networks and their capability to model complex relationships, such as varying drug efficacy based on dosage, beyond simple linear fits.

![Neural Network Squiggle Fit](nn_squiggle.png)

## What is a Neural Network?

Neural networks consist of interconnected layers of nodes or neurons, where each connection represents a parameter (weight or bias) that the network learns during training.

![Connection Values in Neural Networks](connection_value.png)

Initially, these parameters are unknown and are determined through a process called backpropagation, fine-tuning them to fit the training data accurately.

![Parameter Estimation in Neural Networks](unknown_value.png) ![Backpropagation](back_estimated.png)

In the next part of this series, we will delve deeper into how backpropagation works, but for now, let's assume we have a neural network already trained for our dataset.

### Activation Functions

Within the network, some nodes incorporate activation functions, depicted by curved lines, essential for introducing non-linearity into the model.

![Activation Functions](curved_lines.png)

### Hidden Layers

Hidden layers lie between the input and output layers. The configuration of these layers, including the number of nodes and layers, is crucial in designing a neural network.

![Hidden Layers](hidden_layers.png)

### Model Construction

We'll examine how neural networks utilize activation functions in hidden layers to create and combine various shapes, ultimately fitting a complex model to the data, illustrated by a green squiggle in our example.

![Combining Shapes in Neural Networks](add_together.png)

#### Dosage and Efficacy Example

Considering a scenario modeling drug efficacy from dosage, we input dosage values into the network, transforming them through layers and activation functions to predict efficacy.

For the lowest dosage (0), the calculation through one path in the network might look like this:

![Dosage to Activation](from_to.png)

$(\text{Dosage} \times -34.4) + 2.14 = \text{x-axis coordinate}$

$(0 \times -34.4) + 2.14 = 2.14$

![X-Axis Coordinate](x_axis_coordinate.png)

By applying the softplus activation function, we plot a point on the curve for the corresponding dosage, creating a series of points that form a curve as we vary the dosage from 0 to 1.

![Softplus Activation Function](choose_softplus.png) ![Plotting Points](put_dot.png)

The choice of activation function affects the shape of the curve. For instance, using a sigmoid function would yield a different curve.

![Sigmoid Activation Function](choose_sigmoid.png)

As we increase the dosage to 1, we obtain a curve that represents the model's predictions over the dosage range.

![Blue Curve](blue_curve.png)

The range of values from the activation function is relatively narrow, corresponding to the red box in the graph, shaping the curve.

![Narrow Value Range](narrow_range.png)

Scaling and transforming these curves through weights and biases in the network allows us to fit complex models to the data.

![Scaling and Transforming Curves](scale_pos.png) ![New Blue Curve](new_blue_curve.png)

Each node in the hidden layer contributes to the final model by creating unique shapes through the activation function, transformed by the network's parameters.

![Different Parameters and Shapes](diff_para.png) ![Portions of Activation Functions](diff_portion.png) ![Creating New Shapes](exciting_shape.png)

By scaling and adding these curves together, neural networks can approximate complex relationships within the data.

![Adding Curves](add_curve.png) ![Final Model Fit](subtract_curve.png)

### Making Predictions

Given a specific dosage, the neural network can predict its efficacy by mapping it onto the final model curve, indicating the drug's effectiveness.

![Predicting Efficacy](BFSFMs.png)

### Understanding Weights and Biases

Weights and biases are the parameters that neural networks adjust to fit the model to the data, slicing, flipping, and stretching the activation functions into new shapes.

![Weights in Neural Networks](weights.png) ![Biases in Neural Networks](biases.png)

Neural networks can adapt to fit nearly any dataset by manipulating these parameters, showcasing their versatility and power.

![Fitting Complex Data](shift_data.png) ![Theoretical Capability](theory_nn.png)

## Reference:

For a more detailed exploration, watch the related video on [YouTube](https://www.youtube.com/watch?v=CqOfi41LfDw).



# Neural Networks Overview


We will learn about what neural networks do and how they do it.
For example, different dosages will have different efficacy, and we can not just fit a line for the dataset. But even if we have a really complicated dataset like this, a neural network can fit a squiggle to it.

<img src="nn_squiggle.png" alt="nn_squiggle" width="400" height="300"/>

## What is a Neural Network?

A neural network is composed of layers of nodes, with connections between these nodes. The values along each connection represent the parameters of the network, which are estimated during the training process.

<img src="connection_value.png" alt="connection_value" width="400" height="300"/>

For now, just know that these parameter estimates are analogous to the slope and intercept values that we solve for when we fit a straight line to data. 

Initially, the parameter values of a neural network are unknown. These values are determined through a process called backpropagation, which adjusts the parameters to fit the network to the training data.


<img src="unknown_value.png" alt="unknown_value" width="400" height="300"/> <img src="back_estimated.png" alt="back_estimated" width="400" height="300"/>

And we will talk about how backprogagation estimates these parameters in part2 in this series, but for now, just assume that we’ve already fit this neural network to this specific dataset and that means we have already estimated these parameters. 

### Activation Function

Also, you may have noticed that some of the nodes have curved lines inside of them, the curved or bent lines are called activation functions. When you build a neural network, you have to decide which activation function or functions you want to use.

<img src="curved_lines.png" alt="curved_lines" width="400" height="300"/>

### Hidden Layers

The layers between the input and output layers are known as hidden layers. The configuration of these layers—how many there are and how many nodes they contain—is one of the initial design decisions when constructing a neural network.

<img src="hidden_layers.png" alt="hidden_layers" width="400" height="300"/>

### Constructing the Model

Let’s learn how this neural network creates new shapes from the curved or bent lines in the hidden layer and then adds them together to get a green squiggle that fits the data. Note in this neural network, we have one input node which is dosage, one hidden layer with 2 nodes and one output node which is efficacy.


<img src="add_together.png" alt="add_together" width="400" height="300"/>

#### Example: Dosage and Efficacy

Consider a scenario where we model the efficacy of a drug based on its dosage, ranging from 0 (low) to 1 (high). The neural network takes the dosage value, applies transformations through the network's parameters and activation functions, and outputs an efficacy prediction.

Note: To keep the math simple, let’s assume dosages go from 0(low) to 1(high).

The first thing we are going to do is plug the lowest dosage 0 into the neural network. Now, to get from the input node to the top node in the hidden layer, this connection multiplies the dosage by -34.4 and then adds 2.14 and the result is an x-axis coordinate for the activation function.

<img src="from_to.png" alt="from_to" width="400" height="300"/>

$(\text{Dosage} \times -34.4) + 2.14 = \text{x-axis coordinate}$

$(0 \times -34.4) +2.14 = 2.14$

<img src="x_axis_coordinate.png" alt="x_axis_coordinate" width="400" height="300"/>

To get the corresponding y-axis value, we plug 2.14 into the activation function, which, in this case, is the softplus function. Then we put a blue dot at 2.25 for dosage = 0.

<img src="choose_softplus.png" alt="choose_softplus" width="400" height="300"/> <img src="put_dot.png" alt="put_dot" width="400" height="300"/>

Note, if we had chosen the sigmoid curve for the activation function, then we would plug 2.14 into the equation for the sigmoid curve.

<img src="choose_sigmoid.png" alt="choose_sigmoid" width="400" height="300"/>

And if we continue to increase the dosage values all the way to 1(the maximum dosage), we get this blue curve.

<img src="blue_curve.png" alt="blue_curve" width="400" height="300"/>

Note, the full range of dosage values, from 0 to 1 corresponds to this relatively narrow range of values from the activation function. When we plug dosage values, from 0 to 1 into the neural network, and then multiply them by -34.4 and add 2.14, we only get x-axis coordinates that are within the red box and thus, only the corresponding y-axis values in the red box are used to make this new blue curve.

<img src="narrow_range.png" alt="narrow_range" width="400" height="300"/>

Now we scale the y-axis values for the blue curve by -1.30. For example, when dosage = 0, the current y-axis coordinate for the blue curve is 2.25, then 2.25*-1.3 = -2.93 which corresponds to this position on the y-axis. And we end up with a new blue curve.

<img src="scale_pos.png" alt="scale_pos" width="400" height="300"/> <img src="new_blue_curve.png" alt="new_blue_curve" width="400" height="300"/>

Now let’s focus on the connection from the input node to the bottom node in the hidden layer. Remember, for -2.52 and 1.29, these values come from fitting the neural network to the data with back propagation, and we’ll talk about that in part2 in this series.

<img src="remember.png" alt="remember" width="400" height="300"/>

Then same operation as before, we get this orange curve.

<img src="orange_curve.png" alt="orange_curve" width="400" height="300"/>


So we see that fitting a neural network to data gives us different parameter estimates on the connections and results in each node in the hidden layer using different portions of the activation functions to create these new and exciting shapes.

<img src="diff_para.png" alt="diff_para" width="400" height="300"/> <img src="diff_portion.png" alt="diff_portion" width="400" height="300"/> <img src="exciting_shape.png" alt="exciting_shape" width="400" height="300"/>


Now just like before, we scale the y-axis coordinates on the orange curve. Only this time, we scale by a positive number 2.28, and that gives us this new orange curve.

<img src="new_orange_curve.png" alt="new_orange_curve" width="400" height="300"/>


Now the neural network tells us to add the y-axis coordinates from blue curve to the orange curve, and that gives us this green squiggle.


<img src="add_curve.png" alt="add_curve" width="400" height="300"/>


Then finally we subtract 0.58 from the y-axis values on the green squiggle and we have a green squiggle that fits the data.

<img src="subtract_curve.png" alt="subtract_curve" width="400" height="300"/>

### Predictions

Now if someone comes along and says that they are using dosage = 0.5 we can look at the corresponding y-axis coordinate on the green squiggle and see that the dosage will be effective or we can solve for the y-axis coordinate by plugging dosage = 0.5 into the neural network and do the math. And we see that the y-axis coordinate on the green squiggle is 1.03, and since 1.03 is closer to 1 than 0, we will conclude that a dosage = 0.5 is effective.

<img src="BFSFMs.png" alt="BFSFMs" width="400" height="300"/>

### Weights and Biases

The parameters that we multiply are called weights and the parameters that we add are called biases.

<img src="weights.png" alt="weights" width="400" height="300"/> <img src="biases.png" alt="biases" width="400" height="300"/>

Note, this neural network starts with two identical activation functions, but the weights and biases on the connections slice them, flip them and stretch them into new shapes, which are then added together to get a squiggle that is entirely new, and the squiggle is then shifted to fit the data.

<img src="shift_data.png" alt="shift_data" width="400" height="300"/>

In theory, neural networks can fit a green squiggle to just about any dataset, no matter how complicated!

<img src="theory_nn.png" alt="theory_nn" width="400" height="300"/>

## Reference:
- [Watch the video on YouTube](https://www.youtube.com/watch?v=CqOfi41LfDw)
