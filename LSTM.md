# LSTM (Long Short-Term Memory)

LSTM is a type of **Recurrent Neural Network (RNN)** architecture. It was designed to remember long-term dependencies in sequence data, which plain RNNs struggle with due to the vanishing gradient problem.

## Core Concept

**Memory Cell:** The central concept of the LSTM is the “cell state”, or memory cell. It’s a kind of horizontal line that runs through the top of the LSTM’s structure. Information can be written to, read from, or forgotten from this cell state, allowing the LSTM to maintain and modify its memory over time.

## Explain Long Short-Term Memory

**Scenario: Writing a Story**

- **Short-Term Memory:** As you write, you remember the immediate previous events, such as recent details about characters or events. For example, if Sarah picked up a blue umbrella in the last paragraph because it started raining, you remember these details for immediate continuity.
  
- **Long-Term Memory:** This is where past significant events or details are remembered to use when they become relevant again. For instance, you might recall something about a character's background or past to inform a current event in the story.

**“Dependencies” in Languages:** They refer to the relationships between words and how they interact to convey meaning. These can be:
- **Syntactic:** Related to the structure of sentences.
- **Semantic:** Related to meaning.

**Limitation of Memory:** LSTMs have a finite memory, which is updated at every step. They need to decide what information to retain and what to discard, which might sometimes lead to forgetting critical information from the distant past.

## LSTM Architecture

- **Gates:** LSTMs have three main gates:
  - **Forget Gate:** Decides what information should be thrown away or kept.
  - **Input Gate:** Updates the cell state with new information.
  - **Output Gate:** Determines the output based on the cell state and the input.

- **Cell State:** Acts as the memory of the LSTM, carrying both long-term and short-term information. It's a vector that gets updated as new input arrives.

- **Hidden State:** Represents the short-term memory of the LSTM.

The LSTM uses its gates to decide how to update the cell state. For instance, the Forget Gate might decide older information is no longer relevant, while the Input Gate decides the significance of the new information.
