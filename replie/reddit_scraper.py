#!/usr/bin/python
# coding=utf-8
"""
Author: tal
Created on 20/12/2017
All Rights reserved to IDOMOO.INC 2013

"""
import codecs

import praw
import logging

import re

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger = logging.getLogger('prawcore')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

reddit = praw.Reddit(client_id='4LJuNor_6aSiZQ', client_secret="Fga1RhNb_NPpgKvnQCcsM1QbeS0",
                     password='mhpuri1491', user_agent='repliescript',
                     username='mushi8tm')



q_file = codecs.open('questions.txt', 'w', encoding='utf-8')
ans_file = codecs.open('ans.txt', 'w', encoding='utf-8')

matcher = re.compile(r'\[.*?\]')
subred = reddit.subreddit('AskReddit')
try:
    for sub in subred.submissions():
        if len(sub.comments) > 0:
            print("----------------------------------------------")
            title = sub.title.replace('\n', ' ')
            title = re.sub(r'\[.*?\]', '', title).lstrip().rstrip()
            # sub.comments.replace_more(limit=None)
            counter = 0
            subid = sub.id
            print("subId: %s title: %s " % (subid, title))
            for top_level_comment in sub.comments:
                try:
                    if top_level_comment.author is None or top_level_comment.author.name != 'AutoModerator':
                        body = top_level_comment.body.replace('\n', ' ')
                        print("%s) %s " % (counter, body))
                        q_file.write(title + '\n')
                        ans_file.write(body + '\n')
                        counter += 1

                    if counter == 3:
                        break
                except Exception as e:
                    print("Exception : %s , SubID: %s" % (e, subid))
            print("----------------------------------------------")
            q_file.flush()
            ans_file.flush()
except Exception as e:
    print("Exception : %s , SubID: %s" % (e, subid))
finally:
    q_file.close()
    ans_file.close()

