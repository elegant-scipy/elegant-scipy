import evaluate as ev
import numpy as np

def markdown_table(cols, fmt='|%i|%.2f|%.2f|'):
    print('| Month | P(rain)  | P(shine) |\n'
          '| -----:| -------- | -------- |')
    for row in zip(*cols):
        print(fmt % row)

if __name__ == '__main__':
    months = range(1, 13)
    prains = [25, 27, 24, 18, 14, 11, 7, 8, 10, 15, 18, 23]
    prains = [p / 100 for p in prains]
    pshine = [1 - p for p in prains]
    markdown_table([months, prains, pshine])
    probs = np.array([prains, pshine])
    H = - np.sum(probs * np.log2(probs)) * 1/12
    print(H)
    print(ev.split_vi(probs / 12, ignore_x=[], ignore_y=[]))
