import numpy as np 
from bertopic import BERTopic
from umap import UMAP
import click

@click.command()
@click.option('--path_to_train_data')
@click.option('--path_to_save')
def train(path_to_train_data, path_to_save):
    """
    Input: path to np array data of abstracts
    """
    train_data = np.load(path_to_train_data)
    umap_model = UMAP(n_neighbors = 15, 
                      n_components = 5, 
                      min_dist = 0.0,
                      metric = 'cosine', 
                      random_state = 42)

    model = BERTopic(language = "english",
                     umap_model = umap_model,
                     verbose = True)
    topics, probs = model.fit_transform(train_data)
    model.save(path_to_save)
    


if __name__ == "__main__":
    train()