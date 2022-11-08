import numpy as np
import pandas as pd 
from collections import defaultdict
import os
import json


def convert_data_co_csv(path, chunks, all_keys):
    for file in chunks:
        references_dict = defaultdict(list)
        with open(path + file, 'rb') as f:
#             cnt = 0 
            for _ in range(10**7):
                try:
                    element = json.loads(next(f))
# used for monkey-flipping with test5.json, doesn't matter
#                     cnt += 1
#                     if cnt < 10**6:
#                         continue
                    for reference in element.get('references', [np.nan]):
                        for key in list(all_keys):
                            references_dict[key].append(element.get(key, np.nan))

                        authors = element.get('authors')
                        if authors is not None:
                            author_ids = '; '.join([x.get('_id', '') for x in authors if x is not None])
                            author_names = '; '.join([x.get('name', '') for x in authors if x is not None])
                        else:
                            author_ids = np.nan
                            author_names = np.nan
                        references_dict['author_ids'].append(author_ids)
                        references_dict['author_names'].append(author_names)

                        venue = element.get('venue')
                        if venue is not None:
                            venue_id = element['venue'].get('_id', np.nan)
                            venue_name = element['venue'].get('raw', np.nan)
                        else:
                            venue_id = np.nan
                            venue_name = np.nan
                        references_dict['venue_id'].append(venue_id)
                        references_dict['venue_name'].append(venue_name)

                except StopIteration:
                    break

        references_dict['id'] = references_dict.pop('_id')
        data = pd.DataFrame.from_dict(references_dict)
        del references_dict
        
        data = data.loc[data.astype(str).drop_duplicates().index]
        data = data[(data['year'] >= 1940)
                & (data['lang'] == 'en')
                & (~data['title'].isna())
                & (~data['abstract'].isna())
                & (data['abstract'] != '')
                & (~data['references'].isna())]    
        
        name = file.removesuffix('.json')
        data.to_csv(f'all_data/{name}_3.csv', index = False)

        
        
if __name__ == '__main__':
    path = 'data/articles_data/'
#     chunks = os.listdir(path)
#     chunks = ['test1.json', 'test2.json', 'test3.json', 'test4.json']
    # preproc 5 in other way, think about it, for now simply use first 10**6 rows
    # then change script and do it for the ress data
    chunks = ['test5.json']
    all_keys = {'keywords', 'volume', 'year', 'n_citation', 
                'isbn', 'lang', 'venue', 'issue', 'url', 'fos', 
                'references', 'doi', 'page_end', 'title', '_id', 
                'authors', 'pdf', 'page_start', 'abstract', 'issn'}
    
    convert_data_co_csv(path, chunks, all_keys)

    