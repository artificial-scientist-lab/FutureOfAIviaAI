import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from rake_nltk import Metric, Rake
from collections import Counter

if __name__ == "__main__":
    with open('arxiv_data_new.pkl', 'rb') as f: 
        all_titles, all_abstracts, all_dates, all_creator, all_cat, all_id = pickle.load(f)
    
    wnl=WordNetLemmatizer()
    
        
    num_of_abstracts=len(all_titles)
    
    personal_stop_list=['presents','us','show','one','two','three','describes','new','approach','many','introduces','http','also','whose', 'prove','select ','take']
    nltk_stop_list=nltk.corpus.stopwords.words('english')
    full_stop_list=nltk_stop_list+personal_stop_list
    
    
    
    all_key_concepts=[]
    cc=0
    for id_of_abstract in range(num_of_abstracts):
        cc+=1
        if (cc%5000)==0:
            print(str(cc)+'/'+str(num_of_abstracts))
            
        single_string=(all_titles[id_of_abstract]+' '+all_abstracts[id_of_abstract]).replace('\n',' ').replace('-',' ').lower()
        r = Rake(stopwords=full_stop_list,
                 ranking_metric=Metric.WORD_DEGREE,
                 min_length=2)
    
        #print(single_string)
        r.extract_keywords_from_text(single_string)
        ll=r.get_ranked_phrases_with_scores()
    
        for ii in range(len(ll)):        
            if ll[ii][0]>1:
                curr_concept=ll[ii][1]
                if curr_concept.find('.')==-1 and curr_concept.find(',')==-1 and curr_concept.find('~')==-1 and curr_concept.find('http')==-1 and curr_concept.find('&')==-1 and curr_concept.find('(')==-1 and curr_concept.find('$')==-1 and curr_concept.find(' et al')==-1 and curr_concept.find('}')==-1 and curr_concept.find('^')==-1 and curr_concept.find('/')==-1  and curr_concept.find('{')==-1:
                    #print(ll[ii]) 
                    
                    # singularization on the last word
                    curr_concept_split=curr_concept.split(' ') 
                    curr_concept_split[-1]=wnl.lemmatize(curr_concept_split[-1])
                    curr_concept_split[-1]=curr_concept_split[-1].replace('matroids','matroid')
                    curr_concept_singularized=' '.join(curr_concept_split)
                    all_key_concepts.append(curr_concept_singularized)
            
    all_concepts_count=Counter(all_key_concepts).most_common()

full_list=[]
for letter, count in all_concepts_count:
    if len(letter.split(' '))>=3 and count>=3:
        #print(letter, ': ',count)
        full_list.append(letter)
print("3: ",len(full_list))


for letter, count in all_concepts_count:
    if len(letter.split(' '))==2 and count>=6:
        #print(letter, ': ',count)
        full_list.append(letter)
        
print("total: ",len(full_list))

with open("full_concepts_20210320.pkl", "wb") as output_file:
    pickle.dump(full_list, output_file)
    
f = open("full_concepts_20210320.txt", "a")
for ii in range(len(full_list)):
    f.write(full_list[ii]+'\n')
f.close()