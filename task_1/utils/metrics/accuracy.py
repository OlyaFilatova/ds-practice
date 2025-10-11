def accuracy(expected, predicted):
    paired = list(zip(
        [int(label) for label in expected],
        [int(item) for item in predicted]
    ))

    verified = [int(pair[0]) == int(pair[1]) for pair in paired]
    
    accuracy = (len([item for item in verified if item]) / len(verified)) * 100

    return accuracy, paired
