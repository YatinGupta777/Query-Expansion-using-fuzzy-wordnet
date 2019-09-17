from nltk.corpus import wordnet
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
import nltk 
from nltk.tokenize import sent_tokenize 
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import networkx as nx
from textblob import Word
import gensim 
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec 
from gensim.models import FastText



# Load vectors directly from the file
# Google word2vec model
# =============================================================================
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# =============================================================================
# Glove model
# =============================================================================
# glove_input_file = 'glove.6B.100d.txt'
# word2vec_output_file = 'glove.6B.100d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)
# model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# =============================================================================

#FastText
#1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset
# =============================================================================
# model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
# =============================================================================

stop_words=set(stopwords.words("english"))
word_data = []

def stemming(word):
    # stemming of words
    porter = PorterStemmer()
    stemmed = porter.stem(word)     
    return stemmed
   
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

original_queries = []
expanded_queries = []
f = open("topics.txt", "r")
for i in f:
    if '<title>' in i :
        i = i[15:]
        original_queries.append(i)
f.close()

query_number = 1
#original_queries = ["superconductors"]
for q in original_queries:
    
    '''For Test String'''
    
    '''POS TAGGING'''
    print(query_number)
    query_number = query_number + 1
    print (q)
    test_string = q
    test_string = test_string.lower()
    test_string = test_string.replace('\n','')
    ''.join([i for i in test_string if i.isalpha()])
    wordsList = nltk.word_tokenize(test_string) 
    filtered_sentence = [w for w in wordsList if not w in stop_words]
    tagged = nltk.pos_tag(filtered_sentence) 
    #Adjective,Adverb,verb,noun
    tag_list = ["JJ","JJS","JJR","NN","NNS","NNP","NNPS","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ","WRB"]
    x = []
    for i in tagged:
        if(i[1] in tag_list):
            x.append(i[0])
        
    filtered_sentence = x
        
    '''STEMMING'''
    #x = []
    #for i in filtered_sentence:
    #    x.append(stemming(i))
    #filtered_sentence = x
    
    '''Wordnet graph'''
    G=nx.Graph()
    print ("Creating Wordnet graph ")
    for x in filtered_sentence:
        try:
            word = Word(x)
            if(len(word.synsets) > 1):
                w = word.synsets[1]
            else:
                continue
            word_data.append(w.name().partition('.')[0])
        
            G.add_node(w.name().partition('.')[0])
            for h in w.hypernyms():
                #print (h)
                word_data.append(h.name().partition('.')[0])
                G.add_node(h.name().partition('.')[0])
                G.add_edge(w.name().partition('.')[0],h.name().partition('.')[0])
                
                for k in h.hypernyms():
                    #print (h)
                    word_data.append(k.name().partition('.')[0])
                    G.add_node(k.name().partition('.')[0])
                    G.add_edge(h.name().partition('.')[0],k.name().partition('.')[0])
                    
                    for j in k.hypernyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])
                        
                    for j in k.hyponyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])   
                    
                for k in h.hyponyms():
                    #print (h)
                    word_data.append(k.name().partition('.')[0])
                    G.add_node(k.name().partition('.')[0])
                    G.add_edge(h.name().partition('.')[0],k.name().partition('.')[0])    
                    
                    for j in k.hypernyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])
                        
                    for j in k.hyponyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])   
                
            for h in w.hyponyms():
                #print (h)
                word_data.append(h.name().partition('.')[0])
                G.add_node(h.name().partition('.')[0])
                G.add_edge(w.name().partition('.')[0],h.name().partition('.')[0])
                        
                for k in h.hypernyms():
                    #print (h)
                    word_data.append(k.name().partition('.')[0])
                    G.add_node(k.name().partition('.')[0])
                    G.add_edge(h.name().partition('.')[0],k.name().partition('.')[0])
                    
                    for j in k.hypernyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])
                        
                    for j in k.hyponyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])   
                    
                for k in h.hyponyms():
                    #print (h)
                    word_data.append(k.name().partition('.')[0])
                    G.add_node(k.name().partition('.')[0])
                    G.add_edge(h.name().partition('.')[0],k.name().partition('.')[0])    
                    
                    for j in k.hypernyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])
                        
                    for j in k.hyponyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])
                        
            #for h in w.part_holonyms():
             #   print("A")
            #for h in w.part_meronyms():
             #   print("B")
            #for h in w.entailments():
             #   print("C")

        except AttributeError as e: 
            continue
            
    print ("Finding edge weights ")       
    for u,v,a in G.edges(data=True):
        try:
            x = model.similarity(u,v)
            G[u][v]['weight'] = abs(x)
        except KeyError:
             continue
    
    #for u,v,a in G.edges(data=True):
     #   print(u,v,a)
    print ("Calculating centrality measures ")    
    
    bw_centrality = nx.betweenness_centrality(G, normalized=True,weight='weight')
    d_centrality = nx.degree_centrality(G)
    c_centrality = nx.closeness_centrality(G,distance='weight')
    pr = nx.pagerank_numpy(G, alpha=0.9,weight='weight')
    hub,authority=nx.hits_numpy(G)
    hit = {}
    for i in hub:
        if i in authority:
            hit[i] = hub[i]+ authority[i]
    
    avg_bw = 0
    avg_d = 0
    avg_c = 0
    avg_pr = 0
    avg_hit = 0
# =============================================================================
#     avg_hub = 0
#     avg_authority = 0
# =============================================================================
    if(len(bw_centrality) > 0):
        for i in bw_centrality:
            avg_bw += bw_centrality[i]
        
        avg_bw = avg_bw/len(bw_centrality)
        
        for i in d_centrality:
            avg_d += d_centrality[i]
        
        avg_d = avg_d/len(d_centrality)
        
        for i in c_centrality:
            avg_c += c_centrality[i]
        
        avg_c = avg_c/len(c_centrality)
        
        for i in pr:
            avg_pr += pr[i]
        
        avg_pr= avg_pr/len(pr)
    
        for i in hit:
            avg_hit += hit[i]
        
        avg_hit= avg_hit/len(hit)    
# =============================================================================
#     for i in hub:
#         avg_hub += hub[i]
#     
#     avg_hub = avg_hub/len(hub)
#     for i in authority:
#         avg_authority += authority[i]
#     
#     avg_authority = avg_authority/len(authority)
# =============================================================================
    
    print ("Getting Final Query")    
    
    bw_words = []
    d_words = []
    c_words = []
    pr_words = []
    hit_words = []
# =============================================================================
#     hub_words = []
#     authority_words = []
# =============================================================================
    
    count = 10
    x = 0
    #To sort dict in decreasing order
    sorted_authority = (sorted(((value, key) for (key,value) in authority.items()),reverse=True))
    
    for w in sorted(bw_centrality, key=bw_centrality.get, reverse=True):
      bw_words.append(w)
      x = x + 1
      if x >= count:
          break
    x = 0  
    for w in sorted(d_centrality, key=d_centrality.get, reverse=True):
      d_words.append(w)
      x = x + 1
      if x >= count:
          break
    x = 0    
    for w in sorted(c_centrality, key=c_centrality.get, reverse=True):
      c_words.append(w)
      x = x + 1
      if x >= count:
          break  
    x = 0    
    for w in sorted(pr, key=pr.get, reverse=True):
      pr_words.append(w)
      x = x + 1
      if x >= count:
          break
    
    x = 0    
    for w in sorted(hit, key=hit.get, reverse=True):
      hit_words.append(w)
      x = x + 1
      if x >= count:
          break
# =============================================================================
#     x = 0    
#     for w in sorted(hub, key=hub.get, reverse=True):
#       hub_words.append(w)
#       x = x + 1
#       if x >= count:
#           break
#     x = 0    
#     for w in sorted(authority, key=authority.get, reverse=True):
#       authority_words.append(w)
#       x = x + 1
#       if x >= count:
#           break
# =============================================================================
   
    #Itrating thorugh all words      
    
    final_words = []   
    for i in filtered_sentence: 
        final_words.append(i)
        
    for i in bw_centrality:
        count = 0
        if i in filtered_sentence:
            continue
        if i in bw_words:
            count = count + 1
        if i in d_words:
            count = count + 1
        if i in c_words:
            count = count + 1
        if i in pr_words:
            count = count + 1
        if i in hit_words:
            count = count + 1
# =============================================================================
#         if i in hub_words:
#             count = count + 1
#         if i in authority_words:
#             count = count + 1    
# =============================================================================
        
        if count >= 3 :
            final_words.append(i)
           
    final_query = ""
    
    for i in filtered_sentence: 
        final_query += i + " "
    
    for i in final_words:
        if i not in filtered_sentence:
            final_query += i + " "
            
    print("Final Query : " + final_query)        
    expanded_queries.append(final_query)
    
f= open("expanded_queries_NoModel.txt","w+")
for i in expanded_queries:
     f.write(i)
     f.write("\n")
f.close() 
     
# =============================================================================
# print (bw_centrality)
# print (d_centrality)
# print (c_centrality)
# =============================================================================


'''Creating whole graph G'''
# =============================================================================
# pos=nx.spring_layout(G,k=0.2)
# nx.draw(G,pos,with_labels=True)
# labels = nx.get_edge_attributes(G,'weight')
# #to show edge weights
# #nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
# #plt.savefig("path.png")
# plt.show()
# =============================================================================


#print(G.neighbors())
'''Creating small subgraphs for concerned words'''
# =============================================================================
# for idx,val in enumerate(final_words):
# 
#  source = final_words[0]
#  depth = 2 #look for those within length 2.
#  foundset = {key for key in nx.single_source_shortest_path(G,source,cutoff=depth).keys()}
#  H=G.subgraph(foundset)
#  pos=nx.spring_layout(H,k=0.2)
#  plt.figure(2)
#  nx.draw(H,pos,with_labels=True)
#  labels = nx.get_edge_attributes(H,'weight')
#  nx.draw_networkx_edge_labels(H,pos,edge_labels=labels)
# =============================================================================
# =============================================================================
# =============================================================================
# f.close()
# fout.close()
# =============================================================================

# =============================================================================
# nyms = ['hypernyms', 'hyponyms', 'meronyms', 'holonyms', 'part_meronyms', 'sisterm_terms', 'troponyms', 'inherited_hypernyms']
# for x in filtered_sentence:
#     for synset in (wordnet.synsets(x)):
#         for i in nyms:
#             try:
#                 print (getattr(synset, i))
#             except AttributeError as e: 
#                 print (e)
#                 continue
# ============================================================================

#secondary emission electrons positive ion bombardment cathode material substance chemical fiber mineral paper rock gum 
#secondary emission electrons positive ion bombardment cathode material chemical fiber mineral paper rock gum 
#Source
#http://intelligentonlinetools.com/blog/2016/09/05/getting-wordnet-information-and-building-and-building-graph-with-python-and-networkx/
#https://github.com/ellisa1419/Wordnet-Query-Expansion

#http://intelligentonlinetools.com/blog/2016/09/05/getting-wordnet-information-and-building-and-building-graph-with-python-and-networkx/
'''
 ->  Word2Vec (by Google)
     ->  GloVe (by Stanford)
     ->  fastText (by Facebook)
     https://fasttext.cc/docs/en/english-vectors.html
'''