from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

def summarize_essay(essay, no_of_setences):
    # Parse the text
    parser = PlaintextParser.from_string(essay, Tokenizer("english"))
    # Initialize the summarizer
    summarizer = LexRankSummarizer()
    # Summarize the text
    # Number of sentences in the summary
    summary = summarizer(parser.document, no_of_setences)

    short_essay = ""
    for sentence in summary:
        short_essay = short_essay + " "+ str(sentence)

    return short_essay

if __name__ == "__main__":
    text = '''
    Hi Madeleine,I wanted to mention my shift hours at Panera Bread and also take a moment to express my gratitude. As you already know, today marks my last working day with concessions before I graduate.

Here are the details of my shifts at Panera Bread:
- 05/07: 6.5 hours
- 05/12: 10 hours

I appreciate your understanding and support throughout my time here. It has truly been a fantastic experience, filled with learning opportunities and memorable moments. 

I made some good friends during my time here. Your guidance and leadership have played a significant role in making it so enjoyable and I am truly grateful for everything you have done.

While today marks the end of my journey at concessions, I sincerely hope our paths cross again in the future. Thank you once again for being an amazing manager and for all your support.

Take care, and I hope to see you around.

Warm regards,
Piyush
    '''

    summarize_essay(text, 3)