
# coding: utf-8

# In[ ]:

#generate graphs with chunk_size, batch size
import psycopg2
import pandas as pd
import nltk
import re #This is to parse the HTML code from the given text
from sshtunnel import SSHTunnelForwarder #This is to connect to the local Database

def getCharAuthorData(authors, doc, documentTable = 'aman_content', chunk_size = 1000):
    df = pd.DataFrame()
    conn = None
    output = []
    i = 1
    #nltk.download('punkt')
    try:
        
        with SSHTunnelForwarder(("srn01.cs.cityu.edu.hk", 22),
                            ssh_username='stylometry',
                            ssh_password='stylometry',
                            remote_bind_address=('localhost', 5432),
                            local_bind_address=('localhost', 5400)):
            
            conn = psycopg2.connect(user="stylometry", password="stylometry",
                                database="stylometry", host="localhost", port=5400)
            
            
            cur = conn.cursor()
            query = "SELECT author_id, doc_content FROM " + str(documentTable) + " WHERE author_id IN ("
            flag = False
            for auth in authors:
                if not flag:
                    query = query + str(auth)
                    flag = True
                else:
                    query = query + ", " + str(auth)
            query = query + ") AND doc_id NOT IN ("
            flag = False
            for doc_id in doc:
                if not flag:
                    query = query + str(doc_id)
                    flag = True
                else:
                    query = query + ", " + str(doc_id)
            query = query + ") ;"
            cur.execute(query)
            print("Execution completed")
            rows = cur.fetchall()
            
            print("Read completed")
            print("Number of rows: %s" % (len(rows)))
            for row in rows:
                #tokens = nltk.word_tokenize(row[1])
                
              
                temp = re.sub('<[^<]+?>', '', row[1])
                temp = temp.replace("\r\n","")
                temp = temp.replace("\n","") 
                chars = list(temp)
                
                chunk1 = []
                for x in chars:
                    if (i < chunk_size):
                        chunk1.append(x)
                        i += 1
                    else:
                        chunk1.append(x)
                        xx = ''.join(chunk1)
                        xx = str(xx)
                        chunk1 = []
                        output.append([row[0], xx])
                        i = 1
                if len(chunk1) > 0:
                    xx = ''.join(chunk1)
                    xx = str(xx)
                    chunk1 = []
                    output.append([row[0], xx])
                    i = 1

            df = pd.DataFrame(output, columns=["author_id", "doc_content"])
            print(df.dtypes)
            print("Data Frame created: Shape: %s" % (str(df.shape)))

    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        print('Error %s' % e)
        sys.exit(1)

    finally:
        if conn is not None:
            conn.close()
    
    # print df
    return df

def getCharDocData(authors, doc, documentTable = 'aman_content', chunk_size = 1000):
    df = pd.DataFrame()
    conn = None
    output = []
    i = 1
    #nltk.download('punkt')
    try:
        
        with SSHTunnelForwarder(("srn01.cs.cityu.edu.hk", 22),
                            ssh_username='stylometry',
                            ssh_password='stylometry',
                            remote_bind_address=('localhost', 5432),
                            local_bind_address=('localhost', 5400)):
            
            conn = psycopg2.connect(user="stylometry", password="stylometry",
                                database="stylometry", host="localhost", port=5400)
            
            
            cur = conn.cursor()
            query = "SELECT author_id, doc_content FROM " + str(documentTable) + " WHERE"
            query += " doc_id = '" + str(doc) + "' ;"
            
            cur.execute(query)
            print("Execution completed")
            rows = cur.fetchall()
            
            print("Read completed")
            print("Number of rows: %s" % (len(rows)))
            for row in rows:
                #tokens = nltk.word_tokenize(row[1])
                
              
                temp = re.sub('<[^<]+?>', '', row[1])
                temp = temp.replace("\r\n","")
                temp = temp.replace("\n","") 
                chars = list(temp)
                
                chunk1 = []
                for x in chars:
                    if (i < chunk_size):
                        chunk1.append(x)
                        i += 1
                    else:
                        chunk1.append(x)
                        xx = ''.join(chunk1)
                        xx = str(xx)
                        chunk1 = []
                        output.append([row[0], xx])
                        i = 1
                if len(chunk1) > 0:
                    xx = ''.join(chunk1)
                    xx = str(xx)
                    chunk1 = []
                    output.append([row[0], xx])
                    i = 1

            df = pd.DataFrame(output, columns=["author_id", "doc_content"])
            print(df.dtypes)
            print("Data Frame created: Shape: %s" % (str(df.shape)))

    
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        print('Error %s' % e)
        sys.exit(1)

    finally:
        if conn is not None:
            conn.close()
    
    # print df
    return df


# In[ ]:

'''
authors=[123, 80, 75]
doc = 204
df = getCharAuthorData(authors, doc, documentTable = 'aman_content', chunk_size = 1000)
'''

