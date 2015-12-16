import input_data





def load_data():
    mnist = input_data.read_data_sets(".")

    print 'Reading data set complete'

    return mnist

def extract_0_and_7(images, labels):
    array0 = []
    array7 = []

    for	i in range(len(labels)):
	if labels[i] == 0:
            array0.append( images[i] )
	elif labels[i] == 7:
            array7.append( images[i] )

    return (array0, array7)

if __name__ == "__main__":


    datasets = load_data()

    #print datasets.validation.images.shape                                                                                                                                                                                                                                                                                   
    #print datasets.validation.labels.shape                                                                                                                                                                                                                                                                                   
    #print datasets.validation.labels                                                                                                                                                                                                                                                                                         

    """                                                                                                                                                                                                                                                                                                                       
    SIZE = 1000                                                                                                                                                                                                                                                                                                               
    train_images = datasets.train.images[:SIZE]                                                                                                                                                                                                                                                                               
    train_labels = datasets.train.labels[:SIZE]                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                              
    test_images = datasets.test.images[:1000]                                                                                                                                                                                                                                                                                 
    test_labels = datasets.test.labels[:1000]                                                                                                                                                                                                                                                                                 
    """

    train_images = datasets.train.images
    train_labels = datasets.train.labels

    test_images = datasets.test.images
    test_labels = datasets.test.labels

    array0, array7 = extract_0_and_7(train_images, train_labels)

    print "number of 0 = %d, number of 7 = %d" % ( len(array0),	len(array7) )

