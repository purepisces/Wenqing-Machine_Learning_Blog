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
we start by calculating the weight for the leaf on the left, the weight for the left leaf is the total number of people in the leaf, which is 4 divided by the total number of people in both leaves , which is 4+3 = 7. then we multiply that weight by its associated gini impurity 0.375. now we add the weighted impurity for the leaf on the right.

total gini impurity = weighted average of gini impurities for the leaves
= (4/4+3)0.375+(3/4+3)0.444
= 0.405

so the gini impurity for loves popcorn = 0.405
insert popcorn_gini_impurity.png
Likewise, the gini impurity for loves soda is 0.214
insert soda_gini_impurity.png

Now we need to calculate the gini impurity for age. however, because age contains numeric data, and not just yes/no values, calculating the gini impurity is a little more involved.

The first thing we do is sort the rows by age, from lowest value to highest value, then we calculate the average age for all adjacent people. Lastly, we calculate the gini impurity values for each average age.
insert age_gini_impurity.png

For example, to calculate the fini impurity for the first value, we put age < 9.5 in the root.
Then when we calculate the gini impurity for the leaf on the left,which is 
gini impurity for left leaf  = 1 - (the probability of "yes")^2 - (the probability of "no")^2
= 1 - (0/0+1)^2 - (1/0+1)^2
= 0
we got 0 and this makes sense because every single person in this leaf does not love cool as ice. so there is no impurity. 
insert impurity_0.png
Then for gini impurity for right leaf we got 0.5. Now we calculate the weighted average of the two impurities to get the total gini impurity and we get 0.429.
insert 429.png
likewise we calculate the gini impurities for all of the other candidate values and these two candidate threshold, 15 and 44, are tied for the lowest impurity, 0.343. so we can pick either one, in this case, we'll pick 15.
insert tied.png
however, remember that we are comparing gini impurity values for age, loves popcorn and loves soda to decide which feature should be at the very top of the tree. Remember that giniimpurity for them are 0.405, 0.214,0.343 separetaly. Since soda is the lowesr, then we put loves soda at the top of the tree.
insert soda_top.png
Now let's focus on the node on the left. All 4 people that love soda are in this node, 3 of these people love cool as ice and 1 does not, so this node is impure. So let's see if we can reduce the impurity by splitting the people that love soda based on loves popcorn or age. we'll start by asking the 4 people that loves soda if they also loves popcorn. and the total gini impurity for this split is 0.25.
insert love_soda_whether_popcorn.png
Now we test different age threshold just like before, only this time we only consider the ages of people who loves soda and age < 12.5 gives us the lowest impurity, 0 because both leaves have no imurity at all.
insert age12-5.png
now because 0 is less than 0.25, we will use age < 12.5 to split this node into leaves. 
insert split_by_age12_5.png

and note that, the arrow points to are leaves because there is no reason to continue splitting these people into smaller groups.

insert leaves.png

Then there is just one last thing we need to do before we are done building this tree, we need to assign output values for each leaf. generally speaking, the output of a leaf is whatever category that has the most votes. So for the leaf that the majority of the people in this leaf love cool as ice, the output value is loves cool as ice.

insert finish_build_tree.png

Lastthing for some technical details, remember when we built this tree only one person in the original dataset made it to this leaf. Becasuse so few people made it to this leaf, it's hard to have confidence that it will do a great job making predictions with future data. And it's possible that we have overfit the data.

insert one_person.png

In practive, there are two main ways to deal with this problem, one method is called pruning. alternatively, we can put limits on how trees grow, for example, by requiring 3 or more people per leaf. Now we end up with an impure leaf, but also a better sense of the accuracy of our prediction, because we know that only 75% of the people in the leaf loved cool as ice. Note even when a leaf is impure we still need an output value to make a classification and since most of the people in this leaf love cool as ice, that will be the output value.

insert 75.png

also note, when we build a tree, we don't know in advance if it is better to require 3 people er leaf or some other number, so we test different values with something called cross validation and pick the one that works best.
