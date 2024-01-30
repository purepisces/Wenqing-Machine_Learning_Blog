ReLU In Action!!!

Part 1 in neural networks, we started with a simple data set(inside the black box) that showed whether or not different drug dosages were effective against a virus. The low and high dosages were not effective but the medium dosage was effective. Then we talked about how a neural network like this one using the soft plus activation function in the hidden layer can fit a green squiggle to the dataset.
Insert softplus png
Now let’s see what happens if we swap out the soft plus activation function in the hidden layer with one of the most popular activations functions for deep learning and convolutional neural networks——the ReLU activation function, which is short for rectified linear Unit. And as a bonus, because it is common to put an activation function before the final output, we’ll do that too.
Insert ReLU_bonus png

Remember: To keep the math simple, let’s assume Dosages go from 0(low) to 1(high). So if we plug in the lowest dosage, 0, the connection from the input to the top node in the hidden layer multiples the dosage by 1.70 and then adds -0.85, and the result is an x-axis coordinate for the activation function. 

(0x1.70)+-0.85 = -0.85
Insert x-axis png

Now we plug -0.85 into the ReLU activation function, the ReLU activation function output whichever value is larger, 0 or the input value, which in this case is -0.85.
f(-0.85) = max(0,-0.85) = y-axis coordinate = 0

Insert y-axis png

So, let’s put a blue dot at 0 for when dosage = 0.

Insert blue-dot png


And if we continue to increase the dosage values all the way to 1(the maximum dosage), we will get this bent blue line. Then we multiply the y-axis coordinates on the bent blue line by -40.8, and the new bent blue line goes off the screen.

Insert bent-blue-line png
Insert new-bent-blue-line png

Now when we run dosages through the connection to the bottom node in the hidden layer, we get the corresponding y-axis coordinates that go off the screen for this straight orange line. Now we multiply the y-axis coordinates on the straight orange line by 2.7 and we end up with this final straight orange line
Insert bottom-go-off png
Insert straight-orange-line png
Insert final-straight-orange-line png

Now we add the bent blue line and straight orange line together to get this green wedge. Now we add the final bias term, -16 to the y-axis coordinates on the green wedge.

Insert green_wedge png

Lastly, because we included the ReLU activation function right in front of the output, we use the green wedge as its input. For example, the y-axis coordinate for this point on the green wedge is -16 which corresponds to this x-axis coordinate for the ReLU activation function. 
Insert cor_x_axis png

And when we plug that into the ReLU activation function, we get 
f(x) = max(0,x) = y-axis coordinate
f(-16) = max(0,-16) =  y-axis coordinate = 0

And 0 corresponds to this green dot.
Insert cor_green_dot png

And at long las, we end up with this green pointy thing.

Insert green_pointy_thing png

Thus, the ReLU activation function may seem weird, because it is not curvy and the equation is really simple f(x) = max(0,x) = y-axis coordinate
But just like for any other activation function, the weights and biases on the connections slice them, flip them and stretch them into new shapes, which are added together to get an entirely new shape that fits the data.

Insert new_shape png

Some of you may have noticed that the ReLU activation function is bent and not curved, this means that the derivative is not defined where the function is bent, and that’s a problem because gradient descent, which we use to estimate the weights and biases, requires a derivative for all points. However, it’s not a big problem because we can get around this by simply defining the derivative at the bent part to be 0, or 1, it doesn’t really matter.

Insert deri_pro png

