import matplotlib.pyplot as plt

def pianoroll(melody):
    plt.figure(figsize=(25,5))
    plt.scatter(melody['onset'], melody['pitch'], marker='.')