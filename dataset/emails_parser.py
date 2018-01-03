#!/usr/bin/python
# coding=utf-8
"""
Author: tal 
Created on 03/01/2018
All Rights reserved to IDOMOO.INC 2013

"""
import csv
import email
import pandas as pd
import re
import logging
import sys
from elasticsearch import Elasticsearch
import json

maxInt = sys.maxsize
decrement = True

log = logging.getLogger('emails')
log.setLevel(logging.DEBUG)
hand = logging.FileHandler("log_email_parser.log")
hand.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
hand.setLevel(formatter)
log.addHandler(hand)


# Helper functions
def get_text_from_email(msg):
    """To get the content from email objects"""
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload())
    return ''.join(parts)


def split_email_addresses(line):
    """To separate multiple email addresses"""
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs


def parse(input_csv_path):
    # Read the data into a DataFrame
    emails_df = pd.read_csv(input_csv_path, nrows=10)
    log.debug(emails_df.shape)
    emails_df.head()
    # Parse the emails into a list email objects
    messages = list(map(email.message_from_string, emails_df['message']))
    emails_df.drop('message', axis=1, inplace=True)
    # Get fields from parsed email objects
    keys = messages[0].keys()
    for key in keys:
        emails_df[key] = [doc[key] for doc in messages]
    # Parse content from emails
    emails_df['content'] = list(map(get_text_from_email, messages))
    # Split multiple email addresses
    emails_df['From'] = emails_df['From'].map(split_email_addresses)
    emails_df['To'] = emails_df['To'].map(split_email_addresses)

    # Extract the root of 'file' as 'user'
    emails_df['user'] = emails_df['file'].map(lambda x: x.split('/')[0])
    del messages

    emails_df.head()

    # Set index and drop columns with two few values
    emails_df = emails_df.set_index('Message-ID') \
        .drop(['file', 'Mime-Version', 'Content-Type', 'Content-Transfer-Encoding'], axis=1)
    # Parse datetime
    emails_df['Date'] = pd.to_datetime(emails_df['Date'], infer_datetime_format=True)
    emails_df.dtypes

    emails_df.to_csv('emails_output.csv')


def publish_to_es(input_csv_path):
    log.debug("Connecting to ElasticSearch")
    es = Elasticsearch([{'host': '193.106.55.110', 'port': 9200}])
    log.debug("Reading File")
    csv_file = open(input_csv_path, 'r')

    field_names = (
        "Message-ID", "Date", "From", "To", "Subject", "X-From", "X-To", "X-cc", "X-bcc", "X-Folder", "X-Origin",
        "X-FileName", "content", "user")

    try:
        csv_file.readline()  # skip headers
        reader = csv.DictReader(csv_file, field_names)
        rows_processed = 0
        for i, row in enumerate(reader):
            try:
                if row:
                    msg_id = row['Message-ID']
                    if not msg_id:
                        log.debug("No Message ID for  #%s", i)
                        msg_id = None
                    from_raw = row['From']
                    if from_raw:
                        from_m = eval(re.sub(r'frozenset\(|\)', "", from_raw))
                        row['From'] = list(from_m)

                    to_raw = row['To']
                    if to_raw:
                        to = eval(re.sub(r'frozenset\(|\)', "", to_raw))
                        row['To'] = list(to)
                    j = json.dumps(row)
                    es.index(index='enron-emails-1', doc_type='email', id=msg_id, body=j)
                    rows_processed += 1
                    if rows_processed % 1000 == 0:
                        log.info("Processed %s rows", i)
            except TypeError as e:
                log.error("TypeError Caught in Row: #%s Reason: %s", i, repr(e))
            except SyntaxError as e:
                log.error("SyntaxError Caught in Row: #%s Reason: %s", i, repr(e))
            except csv.Error as e:
                log.error("csv.Error Caught in Row: #%s Reason: %s", i, repr(e))
            except KeyError as e:
                log.error("KeyError Caught in Row: #%s Reason: %s", i, repr(e))
            except Exception as e:
                log.error("Exception Caught in Row: #%s Reason: %s", i, repr(e))
    except Exception as e:
        log.error("Exception Caught Reason: %s", repr(e))


if __name__ == '__main__':
    input_csv = '/Users/tal/Downloads/emails.csv'
    # parse(input_csv)
    log.debug("Starting Process")
    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True

    publish_to_es("emails_output.csv")
