import codecs
from elasticsearch import Elasticsearch


def print_search_stats(results):
    print('=' * 80)
    print('Total %d found in %dms' % (results['hits']['total'], results['took']))
    print('-' * 80)


def print_hits(results):
    " Simple utility function to print results of a search query. "
    print_search_stats(results)
    for hit in results['hits']['hits']:
        # get created date for a repo and fallback to authored_date for a commit
        print(
            '/%s/%s (%s): %s' % (hit['_index'], hit['_type'], hit['_id'], hit['_source']['description'].split('\n')[0]))
    print('=' * 80)
    print()


def main():
    es = Elasticsearch([{'host': '193.106.55.110', 'port': 9200}])
    # res = es.search(index="test", doc_type="articles", body={"query": {"match": {"content": "fox"}}})
    res = es.search(index='enron-emails-1', size=10, scroll='2m', doc_type='email',
                    body={
                        "query": {
                            "bool": {
                                "must": [
                                    {
                                        "match_all": {}
                                    },
                                    {
                                        "match_phrase": {
                                            "content": {
                                                "query": "-----Original Message-----"
                                            }
                                        }
                                    }
                                ],
                                "must_not": [
                                    {
                                        "match_phrase": {
                                            "Subject": {
                                                "query": "FW:"
                                            }
                                        }
                                    },
                                    {
                                        "match_phrase": {
                                            "content": {
                                                "query": "---------------------- Forwarded by"
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    })
    sid = res['_scroll_id']
    scroll_size = res['hits']['total']
    from_file = codecs.open('from.txt', 'w', encoding='utf-8')
    to_file = codecs.open('to.txt', 'w', encoding='utf-8')
    u = 0
    while (scroll_size > 0):
        print(u)
        u = u + 1
        res = es.scroll(scroll_id=sid, scroll='2m')
        # Update the scroll ID
        sid = res['_scroll_id']
        # Get the number of results that we returned in the last scroll
        scroll_size = len(res['hits']['hits'])
        print("scroll size: " + str(scroll_size))
        for email in res['hits']['hits']:
            s = email['_source']['content']
            unvalid = ['To:', 'From:', 'Sent:', 'mailto:', '@ENRON', 'atthew.Lenhart@enron.com',
                       'erichardson@sarofim.com', 'cc:', 'Message---', 'RE:']
            arr = s.split('\n')
            arr_rev = []
            i = 1
            # print(arr)
            for st in arr[:]:
                for p in unvalid:
                    if p.lower() in st.lower():
                        arr.remove(st)
                        break
                if st == '':
                    arr.remove(st)
            print(arr)
            sentence = list()
            arr_sen = list()
            for s in arr[:]:
                if '-Original' not in s and 'Subject:' not in s and '- Original' not in s and s:
                    sentence.append(s)
                else:
                    if len(sentence) >= 1:
                        arr_sen.append(' '.join(sentence))
                    sentence = list()
            for k in arr_sen:
                arr_rev.append(arr_sen[-i])
                i = i + 1
            # print(arr_rev)
            i = 0
            num = len(arr_rev) - 1
            while i < num:
                if len(arr_rev[i]) <= 300 and len(arr_rev[i + 1]) <= 300:
                    from_sen = arr_rev[i].strip(' \t\r')
                    from_file.write(from_sen + '\n')
                    to_sen = arr_rev[i + 1].strip(' \t\r')
                    to_file.write(to_sen + '\n')
                i += 1
            from_file.flush()
            to_file.flush()
    from_file.close()
    to_file.close()
    # print("%d documents found" % res['hits']['total'])
    # for doc in res['hits']['hits']:
    #     print("%s) %s" % (doc['_id'], doc['_source']['content']))


if __name__ == "__main__":
    main()
