LSTM(Long Short-Term Memory): 
It is a type of Recurrent Neural Network(RNN) architecture. It was designed to remember long-term dependencies in sequence data, which plain RNNs struggle with due to the vanishing gradient problem.
Memory Cell: The central concept of the LSTM is the “cell state”, or memory cell. It’s a kind of horizontal line that runs through the top of the LSTM’s structure. Information can be written to, read from, or forgotten from this cell state, allowing the LSTM to maintain and modify its memory over time.

Explain Long Short-Term Memory:
Scenario: Writing a Story
Short-Term Memory: 
As you write, you remember the immediate previous events. For instance, if you wrote in the last paragraph that Sarah picked up a blue umbrella because it started raining, in the next paragraph, you’ll remember that she has a blue umbrella and that it’s raining. This is the “short-term” memory in action - it helps in making decisions that are immediately relevant.
Long-Term Memory:
As the story progresses, Sarah meets an old friend named Mike at a cafe. You wrote in the beginning that Sarah and Mike went to school together and shared memories of playing in the rain. Now, many years later, you want to bring up a nostalgic moment where Sarah reminds Mike about those rainy school days. To do this, you need to remember something from way back, not just from the previous paragraph. This is where the “long” in Long Short-Term Memory comes into play. You need a mechanism to remember important events or facts from the distant past to use them when relevant. 
To reiterate, “Long Short-Term Memory” refers to the LSTM’s capability to remember both recent events(short-term) and significant events from the distant past(long-term) when processing sequences.

“Dependencies” in languages refer to the relationships between words and how they interact with each other to convey meaning. These relationships can be syntactic(related to the structure of sentences) or semantic(related to meaning).
Limitation of Memory: LSTMs have a finite memory (the cell state) that is updated at every step. This memory has to encapsulate all the necessary information about the entire input sequence. As new information comes in, some old information might be deemed less relevant and could be forgotten. If, in the very distant future, that information turns out to be critical, an LSTM might struggle since it has discarded it.

LSTM architecture:
Gates: Forget Gate, Input Gate, Output Gate
Cell State: Carry both long-term and short-term information. 
Hidden state: It represents the short-term memory of the LSTM.

Cell State: carry both long-term and short-term information, it is a vector of numbers. The LSTM uses its gates to decide how to update the cell state. 

The Forget Gate may decide that older information (like something about John's childhood friend) is no longer relevant to the current context and reduce its impact.The Input Gate decides how much of the new information (like the fact John moved to Germany) should be stored in the cell state. The Output Gate decides what parts of the cell state should be output as the hidden state for this time step.

