
# coding: utf-8

# # Boston Bombing tweets clustering

# In[11]:


import pandas as pd
import numpy as np
import nltk
import sys

def read_tweets(a):
    path=a
    df=pd.read_json(path,lines=True)
    del df['created_at']
    del df['from_user']
    del df['from_user_id']
    del df['from_user_id_str']
    del df['from_user_name']
    del df['geo']
    del df['id_str']
    del df['iso_language_code']
    del df['location']
    del df['metadata']
    del df['profile_image_url']
    del df['profile_image_url_https']
    del df['source']
    df['text']=df['text'].str.replace("([@])\w+","")
    return df


# In[55]:


def jacquard_dist(a,b,df):
    x=df.loc[df['id'] == a].iloc[-1].to_string()
    y=df.loc[df['id'] == int(b)].iloc[-1].to_string()
    n=0
    w1=set(x.split())-set(nltk.corpus.stopwords.words('english'))
    for word in set(y.split())-set(nltk.corpus.stopwords.words('english')):
        if word in w1:
            n += 1
    Jacquard_dist=1-((n)/(len(w1)+len(y.split())-n))
    return Jacquard_dist


# # Initial seeds

# In[48]:


def get_c(x):
    path=x
    c= open(path, 'r')
    c=[line.strip(',\n') for line in c.readlines()]
    return c



# In[ ]:


def new_cent(c,clusters,df):
    curmin=999
    for i in clusters:
        for j in clusters[i]:
            min=jacquard_dist(j,c[i],df)
            if min<curmin:
                curmin=min
                index=j
            c[i]=index
    return c


# In[ ]:


def calc(a,x,df):
    curmin=99
    for i,h in enumerate(x):
        dist=jacquard_dist(a,h,df)
        if dist<curmin:
            curmin=dist
            index=i
    return index,a


# In[95]:
def sum_squared_errors(c,clusters,df):
    sum=0
    for i in clusters:
        for j in clusters[i]:
            dist=jacquard_dist(j,c[i],df)
            sum=sum+dist**2
    return sum
       

def main(argv):
    clusters={}
    k=int(sys.argv[1])
    for i in range(k):
        clusters[i]=[]
    c=[]
    c=get_c(sys.argv[2])
    df=read_tweets(sys.argv[3])
    op=sys.argv[4]
    while True:
        #clusters= {x:list() for x in range(25)}
        for i in range(len(df)):
            clus_indx,twt_id=calc(df.iloc[i,0],c,df)
            #print(clus_indx,twt_id)
            clusters[clus_indx].append(twt_id)
        old_centroids = c
        c=new_cent(c,clusters,df)
        if c==old_centroids:
            break
    sse=sum_squared_errors(c,clusters,df) 
    with open(op, "a") as file:
        for i in clusters:
            file.write(str(i+1)+ "    ")
            file.writelines(["%s,  " % item  for item in clusters[i]])
            file.write("\n")
            file.write("\n")
            file.write("\n")
        file.write("SSE=  "+str(sse))
         


if __name__ == "__main__":
    main(sys)    
    
    


# In[111]:





# In[93]:




        
            


# In[109]:



        

