import numpy as np

def load_word_embeddings(language):
    print("Loading word embeddings")

    if language == 'english':
        filePath =  "datasets/english/English_Test.tsv"
    elif language == 'spanish':
        filePath = "datasets/spanish/Spanish_Test.tsv"

    f = open(filePath,'r',encoding="utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        try:
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        except ValueError:
            break

    print("Done. ",len(model)," words loaded!")
    return model
