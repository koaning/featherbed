from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import itertools as it 
from skops.io import dump


def featherbed_textrepr(text_stream, dim=300, lite=True, path=None, **kwargs):
    # Make two streams, keep memory footprint low
    stream1, stream2 = it.tee(text_stream)
    
    # Tf/Idf vectorizer can accept generators! 
    tfidf = TfidfVectorizer(**kwargs).fit(stream1)
    X = tfidf.transform(stream2)
    if lite:
        # This makes a pretty big difference
        tfidf.idf_ = tfidf.idf_.astype("float16")

    # Turn the representation into floats 
    svd = TruncatedSVD(n_components=dim, **kwargs).fit(X)
    
    # This makes it much more lightweight to save
    if lite:
        svd.components_ = svd.components_.astype("float16")
    pipe = make_pipeline(tfidf, svd)
    if path:
        # This makes a pretty big difference
        dump(pipe, path)
    return pipe
