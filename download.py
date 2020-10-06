import os, sys, re
import requests
from bs4 import BeautifulSoup

""" Zurnal24ur NEWS ARTICLE WEB SCRAPING
    Please, read README.md for usage.
"""

def clean_html(raw_html):
    """ Remove HTML tags from raw HTML text. Delete unicode character and new line.
    """
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)
    return ' '.join(cleantext.split())
    

if __name__ == '__main__':
    filename = sys.argv[1]
    mode = sys.argv[2]
    
    f = open(filename, 'r')
    urls = f.readlines()
    f.close()

    folder_path = "{}_articles".format(mode)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    print("Downloading articles...")
    article_id = 1
    for line in urls:
        url = line.split('\t')[0]
        # Make a request to get HTML
        res = requests.get(url)
        soup = BeautifulSoup(res.text, features="lxml")

        # Extract article title
        article_title = soup.title.text.split('|', 1)[0]
        
        # Extract article lead text
        article_lead_text = ""
        meta = soup.find("meta", attrs={"name": "description"})
        if (meta is not None):
            article_lead_text = soup.meta["content"].split('|', 1)[0]

        # Remove js code inside article content div
        js_fun = soup.find("div", {"id" : "divInArticle"})
        if (js_fun is not None):
            js_fun.replace_with('')
        
        # Extract article content
        article_content = ""
        content = soup.find("div", {"class" : "article__content"})
        if (content is not None):
            article_content = clean_html(content.text)

        article = [line, article_title, article_lead_text, article_content]
        
        # Save to file
        f = open("{}/{}.txt".format(folder_path, article_id), 'w', encoding="utf8")
        f.writelines(article)
        f.close()

        article_id += 1
    
    print("DONE!")