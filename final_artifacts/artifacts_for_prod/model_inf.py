import pickle
import re
import spacy 
from nltk.corpus import stopwords

class TopicModeling:

    def __init__(self, path_to_model: str):
        self.bert_model = BERTopic.load(path_to_model)
        self.topic_dict = dict(zip(self.bert_model.get_topic_info()['Topic'], 
                                   self.bert_model.get_topic_info()['Name']))
    
    def text_preprocessing(self, text):
        if isinstance(text, str):
            text = [text]
        for idx, txt in enumerate(text):
            regex = re.compile('[A-Za-z]+')
            nlp = spacy.load('en_core_web_sm')
            mystopwords = stopwords.words('english') + ['paper', 'result', 'experiment', 'from', 'subject', 
                                                    're', 'edu', 'use', 'data', 'method', 'based', 
                                                    'new', 'approach', 'also','system', 'model', 
                                                    'present', 'research', 'propose', 'base']
            
            text[idx] = ' '.join(regex.findall(txt))
            doc = nlp(txt)
            text[idx] = ' '.join([token.lemma_ for token in doc])
            text[idx] = ' '.join([token for token in txt.split() if not token in mystopwords])
        
        return text
    
    def score_text(self, text):
        text = self.text_preprocessing(text)
        topics, prob = self.bert_model.transform(text)
        for idx, topic in enumerate(topics):
            topics[idx] = self.topic_dict[topic]
        return topics


if __name__ == "__main__":
    test_text = "The number of states in two-way deterministic finite automata (2DFAs) is considered. It is shown that the state complexity of basic operations is: at least m+ n茂戮驴 o(m+ n) and at most 4m+ n+ 1 for union; at least m+ n茂戮驴 o(m+ n) and at most m+ n+ 1 for intersection; at least nand at most 4nfor complementation; at least $\\Omega(\\frac{m}{n}) + \\frac{2^{\\Omega(n)}}{\\log m}$ and at most $2m^{m+1}\\cdot 2^{n^{n+1}}$ for concatenation; at least $\\frac{1}{n} 2^{\\frac{n}{2}-1}$ and at most $2^{O(n^{n+1})}$ for both star and square; between nand n+ 2 for reversal; exactly 2nfor inverse homomorphism. In each case mand ndenote the number of states in 2DFAs for the arguments."
    list_test_text = [test_text, test_text]
    with open("topic_modeling_pipeline.pkl", "rb") as f:
        m = pickle.load(f)
    print(m.score_text(list_test_text))
    print(m.score_text(test_text))
