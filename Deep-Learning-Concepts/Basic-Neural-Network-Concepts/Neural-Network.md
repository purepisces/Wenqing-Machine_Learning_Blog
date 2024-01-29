Neural networks

We will learn about what neural networks do and how they do it.
For example, different dosages will have different efficacy, and we can not just fit a line for the dataset. But even if we have a really complicated dataset like this, a neural network can fit a squiggle to it.

Insert nn_squiggle png

What is a neural network?
A neural network consists of nodes and connections between the nodes. Note: the numbers along each connection represent parameter values that were estimated when this neural network was fit to the data.
insert connection_value png
For now, just know that these parameter estimates are analogous to the slope and intercept values that we solve for when we fit a straight line to data. Likewise, a neural network starts out with unknown parameter values, that are estimated when we fit the neural network to a dataset using a method called backpropagation. 
Insert unknown_value png
Insert back_estimated png
And we will talk about how backprogagation estimates these parameters in part2 in this series, but for now, just assume that we’ve already fit this neural network to this specific dataset and that means we have already estimated these parameters. Also, you may have noticed that some of the nodes have curved lines inside of them, the curved or bent lines are called activation functions. When you build a neural network, you have to decide which activation function or functions you want to use.
Insert curved_lines png

The layers of nodes between the input and output nodes are called hidden layers. When you build a neural network, one of the first things you do is decide how many hidden layers you want, and how many nodes go into each hidden layer. Although there are rules of thumb for making decisions about the hidden layers, you essentially make a guess and see how well the neural network performs, adding more layers and nodes if needed.
Insert hidden_layers png

Let’s learn how this neural network creates new shapes from the curved or bent lines in the hidden layer and then adds them together to get a green squiggle that fits the data. Note in this neural network, we have one input node which is dosage, one hidden layer with 2 nodes and one output node which is efficacy.
Insert add_together png

Note: To keep the math simple, let’s assume dosages go from 0(low) to 1(high).
The first thing we are going to do is plug the lowest dosage 0 into the neural network. Now, to get from the input node to the top node in the hidden layer, this connection multiplies the dosage by -34.4 and then adds 2.14 and the result is an x-axis coordinate for the activation function.
Insert from_to png
(Dosage x -34.4) + 2.14 = x-axis coordinate 
(0 x -34.4) +2.14 = 2.14  
Insert x_axis_coordinate png
To get the corresponding y-axis value, we plug 2.14 into the activation function, which, in this case, is the softplus function. Then we put a blue dot at 2.25 for dosage = 0.
Insert choose_softplus png
Insert put_dot png
Note, if we had chosen the sigmoid curve for the activation function, then we would plug 2.14 into the equation for the sigmoid curve.
Insert choose_sigmoid png
And if we continue to increase the dosage values all the way to 1(the maximum dosage), we get this blue curve.
Insert blue_curve png
Note, the full range of dosage values, from 0 to 1 corresponds to this relatively narrow range of values from the activation function. When we plug dosage values, from 0 to 1 into the neural network, and then multiply them by -34.4 and add 2.14, we only get x-axis coordinates that are within the red box and thus, only the corresponding y-axis values in the red box are used to make this new blue curve.
Insert narrow_range png
Now we scale the y-axis values for the blue curve by -1.30. For example, when dosage = 0, the current y-axis coordinate for the blue curve is 2.25, then 2.25*-1.3 = -2.93 which corresponds to this position on the y-axis. And we end up with a new blue curve.
Insert scale_pos png
Insert new_blue_curve png
Now let’s focus on the connection from the input node to the bottom node in the hidden layer. Remember, for -2.52 and 1.29, these values come from fitting the neural network to the data with back propagation, and we’ll talk about that in part2 in this series.
Insert remember png
Then same operation as before, we get this orange curve.
Insert orange_curve png

So we see that fitting a neural network to data gives us different parameter estimates on the connections and results in each node in the hidden layer using different portions of the activation functions to create these new and exciting shapes.

Insert diff_para png
Insert diff_portion png
Insert exciting_shape png

Now just like before, we scale the y-axis coordinates on the orange curve. Only this time, we scale by a positive number 2.28, and that gives us this new orange curve.

Insert new_orange_curve png

Now the neural network tells us to add the y-axis coordinates from blue curve to the orange curve, and that gives us this green squiggle.

Insert add_curve png

Then finally we subtract 0.58 from the y-axis values on the green squiggle and we have a green squiggle that fits the data.

Insert subtract_curve png

Now if someone comes along and says that they are using dosage = 0.5 we can look at the corresponding y-axis coordinate on the green squiggle and see that the dosage will be effective or we can solve for the y-axis coordinate by plugging dosage = 0.5 into the neural network and do the math. And we see that the y-axis coordinate on the green squiggle is 1.03, and since 1.03 is closer to 1 than 0, we will conclude that a dosage = 0.5 is effective.

Insert BFSFMs png

The parameters that we multiply are called weights and the parameters that we add are called biases.
Insert weights png
Insert biases png
Note, this neural network starts with two identical activation functions, but the weights and biases on the connections slice them, flip them and stretch them into new shapes, which are then added together to get a squiggle that is entirely new, and the squiggle is then shifted to fit the data.
Insert shift_data png
In theory, neural networks can fit a green squiggle to just about any dataset, no matter how complicated!
Insert theory_nn png
