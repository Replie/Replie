from elasticsearch import Elasticsearch
from dateutil.parser import parse as parse_date


def print_search_stats(results):
    print('=' * 80)
    print('Total %d found in %dms' % (results['hits']['total'], results['took']))
    print('-' * 80)


def print_hits(results):
    " Simple utility function to print results of a search query. "
    print_search_stats(results)
    for hit in results['hits']['hits']:
        # get created date for a repo and fallback to authored_date for a commit
        created_at = parse_date(hit['_source'].get('created_at', hit['_source']['authored_date']))
        print('/%s/%s/%s (%s): %s' % (
            hit['_index'], hit['_type'], hit['_id'],
            created_at.strftime('%Y-%m-%d'),
            hit['_source']['description'].split('\n')[0]))

    print('=' * 80)
    print()


def main():
    es = Elasticsearch([{'host': '193.106.55.110', 'port': 9200}])
    # res = es.search(index="test", doc_type="articles", body={"query": {"match": {"content": "fox"}}})
    res = es.search(index='enron-emails-1', doc_type='email',
                    body={"query": {"match": {"From": "mike.maggi@enron.com"}}, 'sort': [
                        {'Date': {'order': 'desc'}}]})

    print "Found: " + str(res['hits']['total'])

    for email in res['hits']['hits']:
        print email

    # print("%d documents found" % res['hits']['total'])
    # for doc in res['hits']['hits']:
    #     print("%s) %s" % (doc['_id'], doc['_source']['content']))


if __name__ == "__main__":
    main()
