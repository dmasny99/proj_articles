import numpy as np 
from tqdm import tqdm
from bertopic import BERTopic

# custom scripts
from text_preprocessing import text_preprocessing

    
def score_data(model, data, path_to_save):
    topics_dict = dict(zip(model.get_topic_info()['Topic'], model.get_topic_info()['Name']))
    result = []
    for idx, paper in tqdm(enumerate(data)):
        preprocesed_abstract = text_preprocessing(paper[1])
        topics, prob = model.transform(preprocesed_abstract)
        result.append([data[idx][0], topics_dict[topics[0]]])
    result = np.array(result)
    np.save(path_to_save, result)
    
    
    
if __name__ == '__main__':
    
    model = BERTopic.load('bert_model_100k')
    data = np.load('all_data/abstracts.npy', allow_pickle = True)
    score_data(model, data, 'all_data/scored_abstracts_all.npy')
    