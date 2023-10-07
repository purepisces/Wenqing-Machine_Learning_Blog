# Speculyzer: A Trustworthy Neural Program Synthesizer

## Overview

Speculyzer introduces a neural program synthesizer that learns to predict when it can't solve a problem and certifies its correctness with synthesized specifications. The system aims to enhance trust and safety in neural program synthesis.

## Key Features

- **Self-Evaluation**: The method's ability to self-evaluate its solutions is paramount.
- **Trustworthiness**: Emphasizing the importance of trust, especially when deploying such systems in real-world applications like fixing GitHub issues.
- **Certification of Correctness**: The system provides a certificate of the correctness of the synthesized code, similar to how software engineers write test harnesses for new code.

## Methods

### The Approach

1. **Propose Candidate Programs**: The AI tries to come up with possible solutions (code) to a programming problem described in natural language.
2. **Propose Specifications (Specs)**: Alongside the potential solutions, the AI also suggests certain criteria or conditions that a correct solution should meet.

#### Types of Specifications

- **Input-Output Test Cases**: Straightforward examples of expected inputs and outputs.
- **Functional Specifications**: Descriptions of the logical relationships between inputs and outputs.

### Verification Process

The AI checks each solution against its set criteria, providing a form of self-verification.

### Intuition

The goal is to make the AI "check its work", providing solutions and reasons (specs) why it thinks its solution is right.

### Speculyzer in Action

The system synthesizes potential code solutions and generates specifications to validate those solutions. For any given programming task:

1. Uses AI to generate potential code solutions and specs.
2. Checks its solutions against the generated specs.
3. Uses another model to predict if it can solve the problem, which solution is likely correct, and which specs best prove its correctness.

## Results

- Evaluated on the **Mostly Basic Python Problems (MBPP)** and **HumanEval** datasets.
- Achieved the highest `pass@1` rate on HumanEval, surpassing other methods.
- Demonstrated a significant trade-off between precision and recall, with the capability to achieve 100% precision (zero error rate).

## Generalization

The system showcases near-identical performance when trained on one dataset and tested on another, suggesting robust generalization.

## Certifying Correctness

- The system produces a specification that certifies a program's correctness while offering insight into its behavior.
- Specs are ranked based on distinctiveness, making them more informative and aiding users in determining the correctness of a program.

---

**Key Takeaway**: Speculyzer represents a significant step towards trustworthy neural program synthesis, emphasizing precision, user trust, and self-verification. It's designed to be accurate, transparent, and self-verifying, vital qualities for real-world adoption.
