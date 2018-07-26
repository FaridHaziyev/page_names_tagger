#In this Module wikipairs are used and pages are
#created for each pair, and in this pages content 
#of english and german wikipage is also added,

from xml.dom.minidom import parse, parseString
from nltk.corpus import wordnet,stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pdb
import re
import pickle
import multiprocessing as mp

stopWords = set(stopwords.words("english"))
lmtzr = WordNetLemmatizer()

def load_xml_data(data_name):
    wikipairs_data = parse(data_name)
    return wikipairs_data

def get_all_pairs(xml_tree):
    all_pairs = xml_tree.getElementsByTagName("Pair")
    return all_pairs

def get_source(pair):
    source = pair.getElementsByTagName("Source")[0]
    return source

def get_target(pair):
    target = pair.getElementsByTagName("Target")[0]
    return target

def get_page_actual_name(page):
    page_ = page.getElementsByTagName("Actual_Name")[0]
    return page_.firstChild.data

def get_page_name(page, source = False):
    if not source:
        page_ = page.getElementsByTagName("Actual_Name")[1]
    else:
        page_ = page.getElementsByTagName("Actual_Name")[0]

    return page_.firstChild.data

def get_links(page):
    links_child = page.getElementsByTagName("Links")[0].firstChild
    if not links_child:
        return set()
    links = links_child.data
    links = links.replace("\n", " ").replace("\t"," ")
    links = set(re.findall("\w+", links))
    return links

def get_cattegories(page):
    categories_child = page.getElementsByTagName("English_Categories")[0].firstChild
    if not categories_child:
        return set()
    cattegories = categories_child.data
    cattegories = cattegories.replace("\n", " ").replace("\t", " ")
    cattegories = set(re.findall("\w+", cattegories))
    return cattegories

def get_page_context(page):
    links = get_links(page)
    categories = get_links(page)
    
    return links.union(categories)

def get_synset_id(synset):
    return (8-len(str(synset.offset())))*"0" + str(synset.offset()) + "-" + synset.pos()

def get_sense_contexts(synsets):
    sense_contexts = {}  
    for s in synsets:
        hypernyms = s.hypernyms()
        synonyms = [x for x in synsets if x != s]
        sisterhood = []

        for h in hypernyms:
            hyponyms = [x for x in h.hyponyms() if x != s]
            sisterhood.extend(hyponyms)
        
        allsenses = set(hypernyms + synonyms + sisterhood + s.hyponyms() +
                s.member_meronyms() + s.substance_meronyms() + s.part_meronyms() +
                s.member_holonyms() + s.substance_holonyms() + s.part_holonyms()) 
        sense_lemmas = []
        for syn_s in allsenses:
            lemmas = syn_s.lemmas()
            sense_lemmas.extend(lemmas)
        sense_lemmas = set(sense_lemmas)
        sense_lemmas = set(map(lambda x: x.name(), sense_lemmas))
        
        s_context = [x.lower() for x in sense_lemmas if x not in stopWords and len(x) > 3]
       
        sense_contexts[s] = set(s_context)

    return sense_contexts



def map_eng_to_deu(pages, *tagged_pages):
    tagged_source = dict()
    for page in pages:
        page_name = get_page_name(page).lower()
        if page_name in tagged_pages[0]:
            source_name = get_page_name(page, source = True).lower()
            tagged_source[source_name] = tagged_pages[0][page_name]
        elif page_name in tagged_pages[1]:
            source_name = get_page_name(page, source = True).lower()
            tagged_source[source_name] = tagged_pages[1][page_name]
        elif page_name in tagged_pages[2]:
            source_name = get_page_name(page, source = True).lower()
            tagged_source[source_name] = tagged_pages[2][page_name]
 
    return tagged_source


def tag_page_title_first_step(pages, df):
    for page in pages:
        page_name = get_page_name(page).lower()
        synsets = wordnet.synsets(page_name)

        if synsets and len(synsets) == 1:
            df[page_name] = synsets[0].name()


def tag_page_title_second_step(pages, ds, df):
    for page in pages:
        page_name = get_page_name(page).lower()
        synsets = wordnet.synsets(page_name)
                 
        if synsets and len(synsets) == 1:
            continue
        elif synsets:
            links = get_links(page)
            for link in links:
                if link in df:
                    sl = wordnet.synset(df[link])
                    if sl in synsets:
                        ds[page_name] = df[link]
                    break

def tag_page_title_third_step(pages, df, ds, dt):
    for page in pages:
        page_name = get_page_name(page).lower()
        synsets = wordnet.synsets(page_name)
           
        if not synsets:
            continue
        else:
            if page_name in ds or page_name in df:
                continue

            sense_context = get_sense_contexts(synsets)
            page_context = get_page_context(page)
            max_s = [0,None]

            for sense in synsets:
                s_context = sense_context[sense]
                intr = len(set(page_context).intersection(s_context))
                if intr > max_s[0]:
                    max_s[0] = intr
                    max_s[1] = sense.name()

            if max_s[1]:
                dt[page_name] = max_s[1]


if __name__ == "__main__":
    all_pages = []
    manager = mp.Manager()
    df = manager.dict()
    ds = manager.dict()
    dt = manager.dict()
    
    wikipages = load_xml_data("../created_datas/wikipair_de_en.xml")  #to load wikipages
    wikipairs = get_all_pairs(wikipages) #to get all wikipairs
    
    #tag_page_title_first_step(wikipairs,df)
    #tag_page_title_second_step(wikipairs, ds,df)
    #tag_page_title_third_step(wikipairs, df, ds, dt)
 
    #tagged_source = map_eng_to_deu(wikipairs, df,ds,dt)

    n = 80
    p_length = len(wikipairs) // n
    f_threads = []
    s_threads = []
    t_threads = []
    output = dict()
    for i in range(n):
        start = i*p_length
        if i == n - 1:
            end = len(wikipairs)           
            f_threads.append(mp.Process(target=tag_page_title_first_step, 
                                          args = (wikipairs[start:end],df,)))

            s_threads.append(mp.Process(target=tag_page_title_second_step, 
                                          args = (wikipairs[start:end],ds, df,)))

            t_threads.append(mp.Process(target=tag_page_title_third_step, 
                                          args = (wikipairs[start:end],df, ds, dt,)))            
        else:
            end = i*p_length + p_length
            f_threads.append(mp.Process(target=tag_page_title_first_step, 
                                          args = (wikipairs[start:end],df,)))

            s_threads.append(mp.Process(target=tag_page_title_second_step, 
                                          args = (wikipairs[start:end],ds, df,)))

            t_threads.append(mp.Process(target=tag_page_title_third_step, 
                                          args = (wikipairs[start:end],df,ds,dt,)))            

    a = 4
    b = 20
    print("started")
    for x in range(a):
        for y in range(b):
            f_threads[x*b+y].start()
        for z in range(b):
            f_threads[x*b+z].join()

    print("second")
    for x in range(a):
        for y in range(b):
            s_threads[x*b+y].start()
        for z in range(b):
            s_threads[x*b+z].join()
    print("third")
    for x in range(a):
        for y in range(b):
            t_threads[x*b+y].start()
        for z in range(b):
            t_threads[x*b+z].join()

    tagged_source = map_eng_to_deu(wikipairs, df,ds,dt)

    with open("germ_page_names_tagged.pkl", "w") as fw:
        for page_name, synset in tagged_source.items():
            print(page_name + "\t" + get_synset_id(wordnet.synset(synset)), file = fw)



