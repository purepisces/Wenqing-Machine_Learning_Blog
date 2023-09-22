# LSTM (Long Short-Term Memory)

LSTM is a type of **Recurrent Neural Network (RNN)** architecture. It was designed to remember long-term dependencies in sequence data, which plain RNNs struggle with due to the vanishing gradient problem.

## Core Concept

- **Memory Cell:** The central concept of the LSTM is the “cell state”, or memory cell. It’s a kind of horizontal line that runs through the top of the LSTM’s structure. Information can be written to, read from, or forgotten from this cell state, allowing the LSTM to maintain and modify its memory over time.

## Explain Long Short-Term Memory

### Scenario: Writing a Story

- **Short-Term Memory:** 
  - As you write, you remember the immediate previous events. For instance, if you wrote in the last paragraph that Sarah picked up a blue umbrella because it started raining, in the next paragraph, you’ll remember that she has a blue umbrella and that it’s raining. This is the “short-term” memory in action - it helps in making decisions that are immediately relevant.

- **Long-Term Memory:**
  - As the story progresses, Sarah meets an old friend named Mike at a cafe. You wrote in the beginning that Sarah and Mike went to school together and shared memories of playing in the rain. Now, many years later, you want to bring up a nostalgic moment where Sarah reminds Mike about those rainy school days. This is where the “long” in Long Short-Term Memory comes into play.

To reiterate, “Long Short-Term Memory” refers to the LSTM’s capability to remember both recent events (short-term) and significant events from the distant past (long-term) when processing sequences.

- **“Dependencies” in Languages:** They refer to the relationships between words and how they interact with each other to convey meaning. These relationships can be:
  - **Syntactic:** Related to the structure of sentences.
  - **Semantic:** Related to meaning.

- **Limitation of Memory:** LSTMs have a finite memory (the cell state) that is updated at every step. As new information comes in, some old information might be deemed less relevant and could be forgotten.

## LSTM Architecture

- **Gates:** 
  - Forget Gate
  - Input Gate
  - Output Gate

- **Cell State:** 
  - Carries both long-term and short-term information. It is a vector of numbers that the LSTM uses its gates to decide how to update.

- **Hidden State:** 
  - Represents the short-term memory of the LSTM.

For example, the **Forget Gate** might decide older information, like something about John's childhood friend, is no longer relevant. The **Input Gate** decides how much of the new information should be stored in the cell state. The **Output Gate** decides what parts of the cell state should be output as the hidden state for this time step.
