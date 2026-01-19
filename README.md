# Q-Learning Action Game: Dodge the Falling Blocks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pygame](https://img.shields.io/badge/Library-Pygame-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📖 Introduction

This project implements a Reinforcement Learning (RL) agent capable of playing a real-time action game called **"Dodge the Falling Blocks"**.

Unlike static grid-based puzzles, this project challenges the limitations of tabular Q-Learning by applying it to a continuous state space environment. By implementing **State Discretization** and an optimized **Epsilon Decay** strategy, the agent successfully learns to control a paddle to dodge falling enemies, achieving a high survival rate within 300 training episodes.

This project was developed for the course **CDS524 - Machine Learning** (Assignment 1).

## 🎮 Game Demo

<img width="419" height="410" alt="image" src="https://github.com/user-attachments/assets/838ba740-afef-439e-b7a0-abd87e20c170" />

*Figure 1: The game interface showing the Blue Agent dodging Red Enemies.*

## ✨ Features

* **Custom Game Environment**: Built from scratch using `Pygame`.
* **Tabular Q-Learning**: Implemented without external RL libraries.
* **State Discretization**: Converts continuous pixel coordinates into a manageable grid-based state space.
* **Epsilon Decay Strategy**: Optimized exploration-exploitation balance (Decay rate: 0.99).
* **Real-time Visualization**: Watch the agent learn and improve in real-time.

## 🚀 Installation & Usage

### Prerequisites
* Python 3.x
* `pygame`
* `numpy` (optional, for array handling)

### Installation
1.  Clone the repository


2.  Install dependencies:
    ```bash
    pip install pygame numpy
    ```

### Running the Training
To start the game and train the agent
