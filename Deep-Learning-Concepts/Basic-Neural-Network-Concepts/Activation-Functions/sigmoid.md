Sigmoid Activation function

In an artificial neural network, an activation function applies a non-linear transformation to the output of a given layer. One nonlinear activation function called sigmoid maps its inputs to the interval between 0 and 1.

$sigmoid(x) = \frac{e^x}{e^x+1} = \frac{1}{1+e^{-x}}$ 

In the case of neural networks, the x that we are passing would be the pre-activated output from a node in a layer. When a sigmoid is applied to a layer, it is being applied to each of the nodes within that layer. So for a fully connected layer for example we will take the sigmoid of each of the weighted sum of inputs for each node within that layer, then we will get the activation of that node. The activated output of the node before passing that as input onto the next layer.
When using sigmoid as an activation function, we can intuitively think of the output of sigmoid when it is closer to 1 that means it is more activated. So the closer to one, the output is the more activated that node is and the closer to zero the output is the less activated that node is.  So therefore since the output of sigmoid lies between 0 and 1, then using the intuition that we just talked about, we can think about sigmoid as giving us a probability of activation for a given node.

<img src="sigmoid_activation.png" alt="sigmoid_activation" width="400" height="300"/>
