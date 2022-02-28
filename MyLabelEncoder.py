import numpy as np
import copy as copy_module

# alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# labelsTrans = ['C', 'A', 'A', 'S', 'D', 'G', 'A', 'G', 'G', 'T', 'S', 'Y', 'G', 'K', 'L', 'T', 'F']

class MyLabelEncoder():

    def __init__(self, labelsList=None, sort=False):
        self.labels = None
        self.mappingDict = None
        self.reverseMappingDict = None

        if labelsList is not None:
            self.fit(labelsList, sort=sort)


    def fit(self, labelsList, sort=False):
        if self.labels is not None:
            raise Exception('You have already fit labels to this encoder!')
        else:
            self.labels = labelsList.copy()
            if sort: self.labels.sort()
            self.mappingDict = dict((c, i) for i, c in enumerate(self.labels))
            self.reverseMappingDict = dict((i, c) for i, c in enumerate(self.labels))


    def transform(self, labelsTrans):
        integer_encoded = [self.mappingDict[char] for char in labelsTrans]
        return(np.array(integer_encoded))


    def inverse_transform(self, intTrans):
        labels_encoded = [self.reverseMappingDict[int] for int in intTrans]
        return (labels_encoded)

    def getEncoderMappingDict(self):
        return(self.mappingDict)

    def getEncoderReverseMappingDict(self):
        return (self.reverseMappingDict)

    def addLabel(self, newLabel):
        self.labels.append(newLabel)
        newInt = len(self.mappingDict)
        self.mappingDict.update({newLabel: newInt})
        self.reverseMappingDict.update({newInt: newLabel})

    def getLabels(self):
        return (self.labels)

    def getLabelsSet(self):
        return (set(self.labels))

    def copy(self):
        return copy_module.deepcopy(self)
        #
        # new_copy = MyLabelEncoder()
        # new_copy.labels = self.labels
        # new_copy.mappingDict = self.mappingDict
        # new_copy.reverseMappingDict = self.reverseMappingDict
        # return new_copy