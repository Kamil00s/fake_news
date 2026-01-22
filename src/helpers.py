import pandas as pd

def merge_text_title(title,text):

    title = str(title) if title is not None else ""
    text = str(text) if text is not None else ""
    
    return title.strip() + " " + text.strip()
