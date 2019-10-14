import test
import copy

traindata = copy.deepcopy(test.TRAIN_DATA)
testdata = copy.deepcopy(test.test_text)

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

for i in range(7):
    test.test_text = concatByNewLine(getFold(i))
    test.TRAIN_DATA = getNotFold(i)

    test.main()
    print("---------------------------------------------------")
