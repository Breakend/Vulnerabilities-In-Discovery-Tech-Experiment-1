from modAL.models import ActiveLearner
import matplotlib
from matplotlib import pyplot as plt
from functools import partial
import glob
from modAL.uncertainty import entropy_sampling
from modAL.batch import uncertainty_batch_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse.csr import csr_matrix #need this if you want to save tfidf_matrix

init_sample_index = None
texts = []
labels = []
i = 0
files = glob.glob('./data/*.txt')
# iterate over the list getting each file 
for fle in files:
   # open the file and then call .read() to get the text 
   with open(fle) as f:
      text = f.read()
      texts.append(text)
      if 'jvt_email_' in fle:
          if init_sample_index is None:
              init_sample_index = i
          labels.append(1)
      else:
          labels.append(0)
      i += 1

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3),
                     min_df = 0, stop_words = 'english', max_features=3000, sublinear_tf=True)
tfidf_matrix =  tf.fit_transform(texts)
BATCH_SIZE=1
preset_batch = partial(uncertainty_batch_sampling, n_instances=BATCH_SIZE)
X_seed, Y_seed = tfidf_matrix[init_sample_index], [labels[init_sample_index]]
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=preset_batch,
    X_training=X_seed, y_training=Y_seed
)

n_queries = 1000
probas = []
corrects = []
times = []
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

for idx in range(n_queries):
    query_idx, query_instance = learner.query(tfidf_matrix)
    learner.teach(tfidf_matrix[query_idx], np.array(labels)[query_idx])
    probas.append(learner.predict_proba(tfidf_matrix)[:,1])
    corrects.append( (learner.predict(tfidf_matrix) == np.array(labels)))
    times.append(idx)
    ax.scatter(x=[idx] * len(probas[-1][corrects[-1]]),  y=probas[-1][corrects[-1]],  c='g', marker='+', label='Correct')
    ax.scatter(x=[idx] * len(probas[-1][~corrects[-1]]),  y=probas[-1][~corrects[-1]],  c='r', marker='x', label='Incorrect')
    ax.scatter(x=[idx] * len(probas[-1][np.array(labels)==1]),  y=probas[-1][np.array(labels)==1], s=80, facecolors='none', edgecolors='b')

# Plot our classification results.
#ax.legend(loc='lower right')
#ax.set_title("ActiveLearner class predictions (Accuracy: {score:.3f})".format(score=unqueried_score))
plt.show()

