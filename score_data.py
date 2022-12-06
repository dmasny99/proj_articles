import numpy as np 
from tqdm import tqdm
from bertopic import BERTopic
import click

# custom scripts
from text_preprocessing import text_preprocessing

@click.command()
@click.option('--model_path')
@click.option('--input_data_path')
@click.option('--path_to_save')
def score_data(model_path, input_data_path, path_to_save):
    """
    Scores already preprocessed texts
    Input data: python dict {id: preprocessed abstract} serialized in npy format
    """
    model = BERTopic.load(model_path)
    data = np.load(input_data_path, allow_pickle=True).item()
    topics_dict = dict(zip(model.get_topic_info()['Topic'], model.get_topic_info()['Name']))
    values = []
    keys = []
    for key, value in data.items():
        keys.append(key)
        values.append(value)
    topics, prob = model.transform(values)
    result = list(zip(keys, topics))
    np.save(path_to_save, result)
    
    
    
if __name__ == '__main__':
    score_data()
    