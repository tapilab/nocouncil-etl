"""
Summarize transcripts in BOX_PATH/.json to BOX_PATH/.summary

Uses Llama version specified in LLAMA_VERSION.

Skips any transcripts that already have a corresponding .summary file.
"""

from dotenv import load_dotenv
import dspy
import json
import ollama
import os
import pandas as pd
import re

load_dotenv()  # Loads variables from .env into environment
PATH = os.getenv('BOX_PATH')

def get_text(js, start, end, no_speech_thresh=.2):
    txts = []
    for j in js[start:end]:
        if j['no_speech_prob'] < no_speech_thresh:
            txts.append(j['text'])
    return ' '.join(txts)

class ExtractProperNames(dspy.Signature):
    """
    Identify all proper names in this text.
    """
    text: str = dspy.InputField()
    proper_names: list[str] = dspy.OutputField(desc="List of proper names.")

class ExtractOrdinanceNumbers(dspy.Signature):
    """
    Identify all ordinance numbers in this text. 
    The term 'ordinance' should appear near each returned value.
    """
    text: str = dspy.InputField()
    ordinance_numbers: list[str] = dspy.OutputField(desc="List of ordinance numbers.")

class ExtractDocketNumbers(dspy.Signature):
    """
    Identify all docket numbers in this text. 
    The term 'docket' should appear near each returned value.
    """
    text: str = dspy.InputField()
    docket_numbers: list[str] = dspy.OutputField(desc="List of docket numbers.")

class ExtractAddresses(dspy.Signature):
    """
    Identify all street addresses in this text.
    """
    text: str = dspy.InputField()
    street_addresses: list[str] = dspy.OutputField(desc="List of full street addresses.")

class FocusedSummary(dspy.Signature):
    """
    Provide a brief bulleted summary of the following transcript from a New Orleans City Council meeting. 
    
    In your summary, be sure to include all relevant information about:
    - proper names
    - ordinance numbers
    - docket numbers
    - street addresses
    """    
    text: str = dspy.InputField()
    proper_names: list[str] = dspy.InputField()
    ordinance_numbers: list[str] = dspy.InputField()
    docket_numbers: list[str] = dspy.InputField()
    street_addresses: list[str] = dspy.InputField()
    summary: str = dspy.OutputField()

class SummaryOfSummaries(dspy.Signature):
    """
    Provide a detailed summary of this description of a New Orleans City Council Meeting. 
    Cover all the main points in depth.
    """
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()

class MeetingSummarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(FocusedSummary)
        # self.summarize_summaries = dspy.ChainOfThought(SummaryOfSummariesFull)
        self.summarize_summaries = dspy.ChainOfThought(SummaryOfSummaries)
        self.get_names = dspy.ChainOfThought(ExtractProperNames)
        self.get_ordinances = dspy.ChainOfThought(ExtractOrdinanceNumbers)
        self.get_dockets = dspy.ChainOfThought(ExtractDocketNumbers)
        self.get_addresses = dspy.ChainOfThought(ExtractAddresses)

    def forward(self, jsons, focus=None, snippets_per_chunk=100):
        summ = self.summarize
        summsumm = self.summarize_summaries
        summaries = []
        all_proper_names = all_ordinance_numbers = all_docket_numbers = all_street_addresses = []
        for i in range(0, len(jsons), snippets_per_chunk):
            start_js = jsons[i]
            end_js = jsons[min(len(jsons)-1, i+snippets_per_chunk-1)]
            text = get_text(jsons, i, i+snippets_per_chunk, no_speech_thresh=.2)
            proper_names = self.get_names(text=text).proper_names
            ordinance_numbers = self.get_ordinances(text=text).ordinance_numbers
            docket_numbers = self.get_dockets(text=text).docket_numbers
            street_addresses = self.get_addresses(text=text).street_addresses                          
            summary = summ(text=text,
                           proper_names=proper_names,
                           ordinance_numbers=ordinance_numbers,
                           docket_numbers=docket_numbers,
                           street_addresses=street_addresses)     
            all_proper_names.extend(proper_names)
            all_ordinance_numbers.extend(ordinance_numbers)
            all_docket_numbers.extend(docket_numbers)
            all_street_addresses.extend(street_addresses)
            # print(summary.summary)
            summaries.append({'summary': summary.summary,
                              'start_time': start_js['start'],
                              'end_time': end_js['end'],
                              'start_id': start_js['id'],
                              'end_id': end_js['id']})
        result = summsumm(text='\n'.join(s['summary'] for s in summaries))
        summaries.insert(0, {'summary': result.summary,
                              'start_time': jsons[0]['start'],
                              'end_time': jsons[-1]['end'],
                              'start_id': jsons[0]['id'],
                              'end_id': jsons[-1]['id']}
                        )
        return result, pd.DataFrame(summaries)



lm = dspy.LM('ollama_chat/' + os.getenv('LLAMA_VERSION'),
            api_base='http://localhost:11434', 
            api_key='', temperature=0.001, max_tokens=10000)
dspy.configure(lm=lm)
summarizer = MeetingSummarizer()
df = pd.read_json(PATH + 'data.jsonl', orient='records', lines=True)

for _, row in df.iterrows():
    print(row.title, row.date, row.video)
    # if we have a transcript
    if row.video is not None:
        fname = PATH + os.path.basename(row.video)
        json_fname = re.sub('.mp4', '.json', fname)
        summary_fname = re.sub('.mp4', '.summary', fname)
        if not os.path.exists(summary_fname) and os.path.exists(json_fname):
            js = [json.loads(l) for l in open(json_fname) if len(l.strip()) > 0]
            if len(js) == 0:
                print('skipping empty transcript')
            else:
                print('summarizing %d snippets' % len(js))
                _, result = summarizer(js)
                result.to_json(summary_fname, 
                               orient='records',
                               lines=True)

