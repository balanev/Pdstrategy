import axelrod as axl
from axelrod.action import Action
from pybrain.datasets import ClassificationDataSet

def getdata(n=1000,
            ld=10,
            players=[axl.Cooperator(), axl.Defector(), axl.TitForTat(), axl.WinStayLoseShift(), axl.Prober(), axl.Grudger()]):
    ds = ClassificationDataSet(ld*2, nb_classes=len(players))
    j = 2
    k = 2
    while j<n:
        for a in (0, 0.01, 0.02):
            tournament = axl.Tournament(players, turns=ld, repetitions=1, noise=a)
            results = tournament.play(keep_interactions=True)
            print()
            for index_pair, interaction in results.interactions.items():
                player1 = tournament.players[index_pair[0]]
                player2 = tournament.players[index_pair[1]]
                r1 = []
                r2 = []
                for i in interaction[0]:
                    if i == (Action.C, Action.C): r2.append(1), r2.append(1), r1.append('R') # CC R 1
                    if i == (Action.D, Action.D): r2.append(-1), r2.append(-1), r1.append('P') # DD P 4
                    if i == (Action.D, Action.C): r2.append(-1), r2.append(1), r1.append('T') # DC T 2
                    if i == (Action.C, Action.D): r2.append(1), r2.append(-1), r1.append('S') # CD S 3
                if r1 != (ld*['P']) and r1 != ld*['R'] and r1 != ld*['S'] and r1 != ld*['T']:
                    print('%i %i %s (%s) vs %s (%s): %s' % (k, j, player1, index_pair[0], player2, index_pair[1], r1))
                    ds.addSample(r2, (index_pair[0]))
                    j+=1
                k+=1
    print(len(ds))
    return ds

if __name__ == '__main__':

    for lengthdataset in range(10, 31):
        ds = getdata(n=10000, ld=lengthdataset)
        ds.saveToFile(filename='r10000_l' + str(lengthdataset) + '.dat')



