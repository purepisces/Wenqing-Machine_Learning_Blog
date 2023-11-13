R squared 

R squared is a metric of correlation that is easy to compute and intuitive to interpret.

We all know about correlation(regular “R”):
-Correlation values close to 1 or -1 are good and tell you two quantitative variables(i.e. weight and size) are strongly related. -Correlation values close to 0 are lame

Why should we care about R^2?
-R^2 is very similar to its hipper cousin, R, but 
-Interpretation is easier:
-it is not obvious that(regular) R = 0.7 is twice as good as R = 0.5
-However, R^2 = 0.7 is what it looks like, 1.4 times as good as R^2 = 0.5
-It’s easy and intuitive to calculate 
Let’s start with an example, here we’re plotting mouse weight on the y-axis with high weights towards the top and low weights towards the bottom and mouse identification numbers on the x-axis with ID numbers one through seven. We can calculate the mean or average of the mouse weights and plot it as a line that spans the graph. We can calculate the variation of the data around this mean as the sum of the squared differences of the weight for each mouse I where I is an individual mouse represented by a red dot and the mean. The variation of the data = sum(weight for mouse I - mean)^2. The difference between each data point is squared so that points below the mean don’t cancel out points above the mean.

Insert variation of ID graph

Now, what if, instead of ordering our mice by their ID#, we ordered them by their size? All we have done is reorder the data on the X-axis. The mean and variation are the exact same as before. The distances between the dots and the line have not changed(just their order).

Insert variation of Size graph


Here’s the question: Given that we know a mouse’s size, is the mean weight the best way to predict mouse weight? No, we can do way better. All we have to do is fit a line to the data. Now we can predict weight with our line. You tell me you have a large mouse, I can look at my line and make a good guess about the weight.

 <img src="fit-a-line.png" alt="fit a line" width="400" height="300"/>


Here’s another question:

Does the line fit the data better than the mean? If so, how much better? By eye, it looks like the line fits the data better than mean. How do we quantify that difference? R^2.


 <img src="quantify-R-square.png" alt="quantify R square" width="400" height="300"/>

The equation for R squared = Var(mean) - Var(line)/Var(mean)
R squared range from 0 to 1, since the variation around the line will never be greater than the variation around the mean and it will never be less than 0. This division also makes r squared a percentage.
Now we’ll walk through an example where we calculate things one step at a time. The result is 0.81 which means there is 81% less variation around the line than the mean or the size/weight relationship accounts for 81% of the total variation. This means that most of the variation in the data is explained by the size/weight relationship.

<img src="example-r-squared.png" alt="example-r-squared" width="400" height="300"/>

Another example, in this example we’re comparing two possibly uncorrelated variables on the y-axis we have mouse weight again, but on the x-axis we now have time spend sniffing a rock. Then by doing the math we see that R^2 = 6%. Thus there is only 6% less variation around the line than the mean or we can say that the sniff/weight relationship accounts for 6% of the total variation. This means that hardly any of the variation in the data is explained by the sniff/weight relationship.
Now when someone says, “The statistically significant R^2 was 0.9” you can think to yourself “Very good! The relationship between the two variables explains 90% of the variation in the data!” And when someone else says “ The statistically significant R^2 was 0.01…” You can think to yourself… “Dag! Who cares if that relationship is significant, it only accounts for 1% of the variation of the data. Something else must explain the remaining 99%.”

What about plain old R? How is it related to R^2?
R^2 is just the square of R.
Now, when someone says, “the statistically significant R(plain old R) was 0.9…” You can think to yourself… 
“0.9 times 0.9 = 0.81. Very good! The relationship between the two variables explains 81% of the variation in the data!"

And when someone else says…
“The statistically significant R was 0.5…”
You can think to yourself…
“0.5 times 0.5 = 0.25. The relationship accounts for 25% of the variation in the data. That’s good if there are a million other things accounting for the remaining 75%, bad if there is only one thing.”

I like R^2 more than just plain old R because it is easier to interpret. Here’s an example for how much better is R = 0.7 than R = 0.5? 
Well, if we convert those numbers to R^2, we see that:
R^2 = 0.7^2 = 0.5 50% of the original variation is explained
R^2 = 0.5^2 = 0.25 25% of the original variation is explained
With R^2, it is easy to see that the first correlation is twice as good as the second. 
That said R^2 does not indicate the direction of the correlation because squared numbers are never negative.
If the direction of the correlation isn’t obvious, you can say, “ the two variables were positively(or negatively) correlated with R^2 = .…
R^2 main ideas
R^2 is the percentage of variation explained by the relationship between two variables.
If someone gives you a value for plain old R, square it!

