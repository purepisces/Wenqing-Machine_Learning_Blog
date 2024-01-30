<img src="mean_std.png" alt="mean_std" width="400" height="300"/>
> Note: The output layer's neuron count often differs from the input layer. Each layer can have a distinct number of neurons, indicating its 'size' or capacity.
>
## Reference:
- [Watch the video on YouTube](https://youtube.com/watch?v=jTzJ9zjC8nU)
$\text{SSR}(a, b) = \sum\limits_{i=1}^{n} (y_i - ax_i - b)^2$

ReLU In Action!!!

Part 1 in neural networks, we started with a simple data set(inside the black box) that showed whether or not different drug dosages were effective against a virus. The low and high dosages were not effective but the medium dosage was effective. Then we talked about how a neural network like this one using the soft plus activation function in the hidden layer can fit a green squiggle to the dataset.

<img src="softplus.png" alt="softplus" width="400" height="300"/>

Now let’s see what happens if we swap out the soft plus activation function in the hidden layer with one of the most popular activations functions for deep learning and convolutional neural networks——the ReLU activation function, which is short for rectified linear Unit. And as a bonus, because it is common to put an activation function before the final output, we’ll do that too.

<img src="ReLU_bonus.png" alt="ReLU_bonus" width="400" height="300"/>

Remember: To keep the math simple, let’s assume Dosages go from 0(low) to 1(high). So if we plug in the lowest dosage, 0, the connection from the input to the top node in the hidden layer multiples the dosage by 1.70 and then adds -0.85, and the result is an x-axis coordinate for the activation function. 

(0x1.70)+-0.85 = -0.85

<img src="x-axis.png" alt="x-axis" width="400" height="300"/>


Now we plug -0.85 into the ReLU activation function, the ReLU activation function output whichever value is larger, 0 or the input value, which in this case is -0.85.
f(-0.85) = max(0,-0.85) = y-axis coordinate = 0

<img src="y-axis.png" alt="y-axis" width="400" height="300"/>


So, let’s put a blue dot at 0 for when dosage = 0.

<img src="blue-dot.png" alt="blue-dot" width="400" height="300"/>

And if we continue to increase the dosage values all the way to 1(the maximum dosage), we will get this bent blue line. Then we multiply the y-axis coordinates on the bent blue line by -40.8, and the new bent blue line goes off the screen.

<img src="bent-blue-line.png" alt="bent-blue-line" width="400" height="300"/>

<img src="new-bent-blue-line.png" alt="new-bent-blue-line" width="400" height="300"/>


Now when we run dosages through the connection to the bottom node in the hidden layer, we get the corresponding y-axis coordinates that go off the screen for this straight orange line. Now we multiply the y-axis coordinates on the straight orange line by 2.7 and we end up with this final straight orange line

<img src="bottom-go-off.png" alt="bottom-go-off" width="400" height="300"/>

<img src="straight-orange-line.png" alt="straight-orange-line" width="400" height="300"/>

<img src="final-straight-orange-line.png" alt="final-straight-orange-line" width="400" height="300"/>


Now we add the bent blue line and straight orange line together to get this green wedge. Now we add the final bias term, -16 to the y-axis coordinates on the green wedge.

<img src="green_wedge.png" alt="green_wedge" width="400" height="300"/>

Lastly, because we included the ReLU activation function right in front of the output, we use the green wedge as its input. For example, the y-axis coordinate for this point on the green wedge is -16 which corresponds to this x-axis coordinate for the ReLU activation function. 

<img src="cor_x_axis.png" alt="cor_x_axis" width="400" height="300"/>

And when we plug that into the ReLU activation function, we get 
f(x) = max(0,x) = y-axis coordinate
f(-16) = max(0,-16) =  y-axis coordinate = 0

And 0 corresponds to this green dot.

<img src="cor_green_dot.png" alt="cor_green_dot" width="400" height="300"/>

And at long las, we end up with this green pointy thing.

<img src="green_pointy_thing.png" alt="green_pointy_thing" width="400" height="300"/>

Thus, the ReLU activation function may seem weird, because it is not curvy and the equation is really simple f(x) = max(0,x) = y-axis coordinate

But just like for any other activation function, the weights and biases on the connections slice them, flip them and stretch them into new shapes, which are added together to get an entirely new shape that fits the data.

<img src="new_shape.png" alt="new_shape" width="400" height="300"/>

Some of you may have noticed that the ReLU activation function is bent and not curved, this means that the derivative is not defined where the function is bent, and that’s a problem because gradient descent, which we use to estimate the weights and biases, requires a derivative for all points. However, it’s not a big problem because we can get around this by simply defining the derivative at the bent part to be 0, or 1, it doesn’t really matter.

<img src="deri_pro.png" alt="deri_pro" width="400" height="300"/>


