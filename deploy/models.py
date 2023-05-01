import pickle

filename = 'final models/count_victorizer.sav'
count_victorizer = pickle.load(open(filename, 'rb'))

filename = 'final models/tfidf_transformer.sav'
transformer = pickle.load(open(filename, 'rb'))


filename = 'final models/model_lr.sav'
model_lr = pickle.load(open(filename, 'rb'))

filename = 'final models/svc_model.sav'
svc_model = pickle.load(open(filename, 'rb'))

filename = 'final models/tree_final_model.sav'
tree_model = pickle.load(open(filename, 'rb'))

filename = 'final models/rfc_model.sav'
rfc_model = pickle.load(open(filename, 'rb'))
