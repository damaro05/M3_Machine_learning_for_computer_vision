import cPickle

def load_input_metadata(ffilenames, flabels):
    images_filenames = cPickle.load(open(ffilenames, 'r'))
    labels = cPickle.load(open(flabels, 'r'))
    return (images_filenames, labels)
