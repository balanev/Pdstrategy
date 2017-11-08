import os
import json
import matplotlib.pyplot as plt

files = os.listdir(".")
for fil in files:
  if fil.endswith(".dta"):
    print(fil)
    with open(fil, 'r') as fp:
         edata = json.load(fp)
    ldta = len(edata)+1
    gtr = []
    gts = []
    for dta in edata:
        gtr.append(dta[0])
        gts.append(dta[1])
    ax = plt.subplot(111)
    plt.plot(range(1, ldta), gtr, 'r', label='Train error')
    plt.plot(range(1, ldta), gts, 'g', label='Test error')
    ax.legend()
    plt.xlabel('Epoch (n)')
    plt.ylabel('Error (%)')
    plt.title('Neural Network Error Trend')
    plt.xlim([0, ldta])
    plt.ylim([0, 21])
    plt.yticks(range(1, 21))
    plt.xticks(range(0, ldta, 10))
    plt.grid(b=True, which='major', color='gray', linestyle='-')
    fig = plt.gcf()
    fig.set_size_inches(7.5, 5)
    plt.savefig(fil.split(".")[0], dpi=300)
    plt.close()
