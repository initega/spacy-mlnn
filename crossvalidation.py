import our_ner_trainer
import copy

traindata = copy.deepcopy(our_ner_trainer.TRAIN_DATA)
testdata = copy.deepcopy(our_ner_trainer.test_text)

def getFold(fold):
    return [traindata[i][0] for i in range(len(traindata)) if i%7 == fold]

def getNotFold(fold):
    return [traindata[i] for i in range(len(traindata)) if i%7 != fold]

def concatByNewLine(strings):
    res = ""
    if len(strings) > 0:
        res = strings[0]
        for s in strings[1:]:
            res += "\n" + s
    return res

for i in range(1):
    our_ner_trainer.test_text = concatByNewLine(getFold(i))
    our_ner_trainer.TRAIN_DATA = getNotFold(i)

    our_ner_trainer.main()
    print("---------------------------------------------------")
