from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
from pybrain.structure import LinearLayer
from pybrain.structure import SigmoidLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.validation import ModuleValidator
from pybrain.tools.validation import Validator
from pybrain.rl.environments.task import Task
from multiprocessing import Pool as ThreadPool
import json

# однослойный перцептрон
def trainet(data, nhide=8, epo = 10, wd = .1, fn=''):
    alldata = data
    tstdata_temp, trndata_temp = alldata.splitWithProportion(0.5)

    tstdata = ClassificationDataSet(alldata.indim, nb_classes=alldata.nClasses)
    for n in range(0, tstdata_temp.getLength()):
        tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

    trndata = ClassificationDataSet(alldata.indim, nb_classes=alldata.nClasses)
    for n in range(0, trndata_temp.getLength()):
        trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

    tstdata._convertToOneOfMany()
    trndata._convertToOneOfMany()

    net = buildNetwork(trndata.indim, nhide, trndata.outdim, hiddenclass=TanhLayer, bias=True)
    net.randomize()
    trainer = BackpropTrainer(net, dataset=trndata, verbose=True, weightdecay=wd, momentum=0.1)
    edata=[]
    msedata=[]
    for i in range(epo):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(), trndata['class'])
        tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
        tod = trainer.testOnData(verbose=False)
        print("epoch: %4d" % trainer.totalepochs,
              "  train error: %5.2f%%" % trnresult,
              "  test error: %5.2f%%" % tstresult,
              "  layers: ", nhide,
              "  N_tourn: ", alldata.indim/2)
        edata.append([trnresult, tstresult])
        msedata.append([i, tod])
    with open(fn+".dta", 'w') as fp:
        json.dump(edata, fp)
    with open(fn+".mse", 'w') as fp:
        json.dump(msedata, fp)
    return net

def fname(n=1000, lg=10, nhide=6, epo=10, wd=.1):
    return 'net_'+str(n)+'_'+str(lg)+'_'+str(nhide)+'_'+str(epo)+'_'+'{}'.format(wd)[2:]

def genet(lst):
    n, lg, nhide, epo, wd = lst
    ds = ClassificationDataSet.loadFromFile('r10000_l' + str(lg) + '.dat')
    fna = fname(n, lg, nhide, epo, wd)
    nt = trainet(ds, nhide, epo, wd, fn=fna)
    NetworkWriter.writeToFile(nt, filename=fna + '.xml')

if __name__ == '__main__':

    epoch = 150
    data = []
    for lenghtGame in range(10, 31):
        for numberHide in range(6, 14):
            data.append([10000, lenghtGame, numberHide, epoch, 0.00001])
    pool = ThreadPool(3)
    results = pool.map(genet, data)
    pool.close()
    pool.join()
