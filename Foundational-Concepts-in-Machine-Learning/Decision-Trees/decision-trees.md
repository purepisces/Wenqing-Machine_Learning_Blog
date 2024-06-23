In general, a decision tree makes a statement and then makes a decision based on whether or not that statement is true or false.
when a decision tree classifies things into categories, it is called classification tree. And when a decision tree predicts numeric values, it's called a regression tree.

insert classification_tree.png insert regression_tree.png

The very top of the tree is called the root node or just the root. 

insert root_node.png

These are called internal nodes, or branches. branches have arros pointing to them, and they have arrows pointing away from them.
insert internal_node.png

Lastly, these are called leaf nodes or just leaves. Leaves have arrows pointing to them, but there are no arrows pointing away from them.
insert leaf_node.png


because these three leaves all contain a mixture of people who do and do not love cool as ice, they are called **impure**.
insert impure.png

There are several ways to quantify the impurity of the leaves, one of the most popular methods is called gini impurity, but there are also fancy soundin methods like entropy and information gain. however, numerically, the methods are all quite similar, so we will focus on gini impurity since not only is it very popular I think it is the most straightforward.

Let's start by calculating the gini impurity for loves popcorn. To calculate the gini impurity for loves popcorn, we start by caculating the gini impurity for the individual leaves.
gini impurity for left leaf  = 1 - (the probability of "yes")^2 - (the probability of "no")^2
= 1 - (1/1+3)^2 - (3/1+3)^2
= 0.375

gini impurity for right leaf  = 1 - (the probability of "yes")^2 - (the probability of "no")^2
= 1 - (2/2+1)^2 - (1/2+1)^2
= 0.444

now because the leaf on the left has 4 people in it, and the leaf on the right only has 3 people in it, the leaves do not represent the same nuber of people. 
Thus the total gini impurity = weighted average of gini impurities for the leaves.
we start by calculating the weight for the leaf on the left
