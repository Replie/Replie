#!/usr/bin/python
# coding=utf-8
"""
Author: tal
Created on 20/12/2017
All Rights reserved to IDOMOO.INC 2013

"""
import codecs
import json

import praw
import logging

import re

import time

from elasticsearch import Elasticsearch

log = logging.getLogger('reddit')
log.setLevel(logging.DEBUG)
hand = logging.FileHandler("log_reddit.log")
hand.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
hand.setFormatter(formatter)
log.addHandler(hand)

reddit = praw.Reddit(client_id='4LJuNor_6aSiZQ', client_secret="Fga1RhNb_NPpgKvnQCcsM1QbeS0",
                     password='mhpuri1491', user_agent='repliescript',
                     username='mushi8tm')

q_file = codecs.open('questions.txt', 'w', encoding='utf-8')
ans_file = codecs.open('ans.txt', 'w', encoding='utf-8')

matcher = re.compile(r'\[.*?\]')
subred = reddit.subreddit('AskReddit')

log.debug("Connecting to ElasticSearch")
es = Elasticsearch([{'host': '193.106.55.110', 'port': 9200}])
start_time = 1483228800
end_time = 1514678400

while True:
    try:
        for sub in subred.submissions(start=start_time,end=end_time):
            if len(sub.comments) > 0:
                log.info("----------------------------------------------")
                title = sub.title.replace('\n', ' ')
                title = re.sub(r'\[.*?\]', '', title).lstrip().rstrip()
                # sub.comments.replace_more(limit=None)
                counter = 0
                subid = sub.id
                log.info("subId: %s title: %s " % (subid, title))
                did_write = False
                j = dict()
                j['subId'] = subid
                j['title'] = title
                for top_level_comment in sub.comments:
                    try:
                        if top_level_comment.author is None or top_level_comment.author.name != 'AutoModerator':
                            body = top_level_comment.body.replace('\n', ' ')
                            log.info("%s) %s " % (counter, body))
                            q_file.write(title + '\n')
                            ans_file.write(body + '\n')
                            j['ans_{0}'.format(counter + 1)] = body
                            did_write = True
                            counter += 1

                        if counter == 3:
                            break

                    except Exception as e:
                        log.info("Exception : %s , SubID: %s" % (e, subid))
                if did_write is True:
                    q_file.flush()
                    ans_file.flush()
                    es.index(index='reddit-1', doc_type='reddit', id=subid, body=json.dumps(j))
                log.info("----------------------------------------------")
    except Exception as e:
        log.info("Exception : %s , SubID: %s" % (e, subid))
        time.sleep(60)
