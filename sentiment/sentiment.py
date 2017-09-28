import subprocess
import shlex
import os

class SentimentAnalysis:
    def __init__(self, path = None):
        self.java_cmd = 'java -jar ' + path + 'SentiStrength.jar stdin sentidata ' + path + 'SentStrength_Data_Sept2011/'

    def rate_sentiment(self, sentiString):
        p = subprocess.Popen(shlex.split(self.java_cmd), stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        sentiString = sentiString.replace('\n', ' ')
        stdout_text, stderr_text = p.communicate(bytes(sentiString.replace(" ","+").encode('utf-8')))
        #stdout_text = stdout_text.strip().replace("\t","")
        #stdout_text = bytes(stdout_text, 'utf-8').rstrip().replace("\t","")
        return stdout_text[:-1]