# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Please note that:
1. You need to first apply for a Google Search API key at https://serpapi.com/,
   and replace the 'your google keys' below before you can use it.
2. The service for searching arxiv and obtaining paper contents is relatively simple. 
   If there are any bugs or improvement suggestions, you can submit pull requests.
   We would greatly appreciate and look forward to your contributions!!
"""
import os
import re
import bs4
import json
import arxiv
import urllib
import zipfile
import warnings
import requests
from datetime   import datetime
warnings.simplefilter("always")

GOOGLE_KEY = os.getenv("GOOGLE_SERPER_KEY", "your google keys")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
arxiv_client = arxiv.Client(delay_seconds = 0.05)
id2paper     = json.load(open("data/paper_database/id2paper.json"))
paper_db     = zipfile.ZipFile("data/paper_database/cs_paper_2nd.zip", "r")

def google_search_arxiv_id(query, num=10, end_date=None):
    url = "https://google.serper.dev/search"

    search_query = f"{query} site:arxiv.org"
    if end_date:
        try:
            end_date = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
            search_query = f"{query} before:{end_date} site:arxiv.org"
        except:
            search_query = f"{query} site:arxiv.org"
    
    payload = json.dumps({
        "q": search_query, 
        "num": num, 
        "page": 1, 
    })

    headers = {
        'X-API-KEY': GOOGLE_KEY,
        'Content-Type': 'application/json'
    }
    assert headers['X-API-KEY'] != 'your google keys', "add your google search key!!!"

    for _ in range(3):
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            if response.status_code == 200:
                results = json.loads(response.text)
                arxiv_id_list = []
                for paper in results['organic']:
                    if re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]):
                        arxiv_id = re.search(r'arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)', paper["link"]).group(1)
                        arxiv_id_list.append(arxiv_id)
                return list(set(arxiv_id_list))
        except:
            warnings.warn(f"google search failed, query: {query}")
            continue
    return []

def _abstract_from_inverted_index(inverted_index):
    if not inverted_index:
        return ""
    words = []
    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))
    return " ".join(word for _, word in sorted(words))

def _extract_arxiv_id_from_openalex_work(work):
    candidates = []
    ids = work.get("ids") or {}
    candidates.extend(str(value) for value in ids.values() if value)
    open_access = work.get("open_access") or {}
    candidates.extend(
        str(open_access.get(key, ""))
        for key in ["oa_url", "landing_page_url", "pdf_url"]
    )
    primary_location = work.get("primary_location") or {}
    candidates.extend(
        str(primary_location.get(key, ""))
        for key in ["landing_page_url", "pdf_url"]
    )
    for location in work.get("locations") or []:
        candidates.extend(
            str(location.get(key, ""))
            for key in ["landing_page_url", "pdf_url"]
        )

    for candidate in candidates:
        match = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", candidate)
        if match:
            return match.group(1).split("v")[0]
    return ""

def openalex_search_papers(query, num=10, end_date=None):
    """Search OpenAlex and return paper dicts compatible with PaperAgent."""
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "sort": "cited_by_count:desc",
        "per-page": min(max(num * 3, num), 50),
        "page": 1,
    }
    filters = ["open_access.is_oa:true"]
    if end_date:
        try:
            year = datetime.strptime(end_date, "%Y%m%d").year
            filters.append(f"publication_year:<{year}")
        except Exception:
            pass
    if filters:
        params["filter"] = ",".join(filters)
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    papers = []
    seen = set()
    for _ in range(3):
        try:
            response = requests.get(base_url, params=params, timeout=20)
            if response.status_code != 200:
                warnings.warn(
                    f"openalex search failed: {response.status_code}, query: {query}"
                )
                continue
            for work in response.json().get("results", []):
                title = work.get("title") or ""
                abstract = _abstract_from_inverted_index(
                    work.get("abstract_inverted_index")
                )
                if not title or not abstract:
                    continue

                arxiv_id = _extract_arxiv_id_from_openalex_work(work)
                if arxiv_id:
                    paper = search_paper_by_arxiv_id(arxiv_id)
                    if paper is not None:
                        paper["source"] = paper["source"] + "+OpenAlex"
                    else:
                        paper = {
                            "arxiv_id": arxiv_id,
                            "title": title.replace("\n", " "),
                            "abstract": abstract.replace("\n", " "),
                            "sections": "",
                            "source": "SearchFrom:openalex_arxiv",
                        }
                else:
                    openalex_id = (work.get("id") or "").split("/")[-1]
                    if not openalex_id:
                        openalex_id = keep_letters(title)[:80]
                    paper = {
                        "arxiv_id": f"openalex:{openalex_id}",
                        "title": title.replace("\n", " "),
                        "abstract": abstract.replace("\n", " "),
                        "sections": "",
                        "source": "SearchFrom:openalex",
                    }

                if paper["arxiv_id"] in seen:
                    continue
                seen.add(paper["arxiv_id"])
                papers.append(paper)
                if len(papers) >= num:
                    return papers
            return papers
        except requests.RequestException as e:
            warnings.warn(f"openalex search error: {e}, query: {query}")
    return papers

def search_papers_by_backend(query, num=10, end_date=None, backend="google"):
    backend = backend.lower()
    if backend == "openalex":
        return openalex_search_papers(query, num, end_date)

    papers = []
    seen = set()
    if backend == "hybrid":
        for paper in openalex_search_papers(query, num, end_date):
            papers.append(paper)
            seen.add(paper["arxiv_id"])
            if len(papers) >= num:
                return papers

    arxiv_ids = google_search_arxiv_id(query, num, end_date)
    for arxiv_id in arxiv_ids:
        arxiv_id = arxiv_id.split("v")[0]
        if arxiv_id in seen:
            continue
        paper = search_paper_by_arxiv_id(arxiv_id)
        if paper is not None:
            papers.append(paper)
            seen.add(arxiv_id)
        if len(papers) >= num:
            break
    return papers

def parse_metadata(metas):
    """
    Parse concatenated metadata string into authors, title, and journal.
    """
    # Get and clean metas
    metas = [item.replace('\n', ' ') for item in metas]
    meta_string = ' '.join(metas)
    
    authors, title, journal = "", "", ""
        
    if len(metas) == 3: # author / title / journal
        authors, title, journal = metas
    else:
        # Remove the year suffix (e.g., 2022a) from the metadata string
        meta_string = re.sub(r'\.\s\d{4}[a-z]?\.', '.', meta_string)
        # Regular expression to match the pattern
        regex = r"^(.*?\.\s)(.*?)(\.\s.*|$)"
        match = re.match(regex, meta_string, re.DOTALL)
        if match:
            authors = match.group(1).strip() if match.group(1) else ""
            title = match.group(2).strip() if match.group(2) else ""
            journal = match.group(3).strip() if match.group(3) else ""

            if journal.startswith('. '):
                journal = journal[2:]

    return {
        "meta_list": metas, 
        "meta_string": meta_string, 
        "authors": authors,
        "title": title,
        "journal": journal
    }

def create_dict_for_citation(ul_element):
    citation_dict, futures, id_attrs = {}, [], []
    for li in ul_element.find_all("li", recursive=False):
        id_attr = li['id']
        metas = [x.text.strip() for x in li.find_all('span', class_='ltx_bibblock')]
        id_attrs.append(id_attr)
        futures.append(parse_metadata(metas))
    results = list(zip(id_attrs, futures))
    citation_dict = dict(results)
    return citation_dict

def generate_full_toc(soup):
    toc = []
    stack = [(0, toc)]
    
    # Mapping of heading tags to their levels
    heading_tags = {'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5}
    
    for tag in soup.find_all(heading_tags.keys()):
        level = heading_tags[tag.name]
        title = tag.get_text()
        
        # Ensure the stack has the correct level
        while stack and stack[-1][0] >= level:
            stack.pop()
        
        current_level = stack[-1][1]

        # Find the nearest enclosing section with an id
        section = tag.find_parent('section', id=True)
        section_id = section.get('id') if section else None
        
        # Create the new entry
        new_entry = {'title': title, 'id': section_id, 'subsections': []}
        
        current_level.append(new_entry)
        stack.append((level, new_entry['subsections']))
    
    return toc

def parse_text(local_text, tag):
    ignore_tags = ['a', 'figure', 'center', 'caption', 'td', 'h1', 'h2', 'h3', 'h4']
    # latexmlc
    ignore_tags += ['sup']
    max_math_length = 300000

    for child in tag.children:
        child_type = type(child)
        if child_type == bs4.element.NavigableString:
                txt = child.get_text()
                local_text.append(txt)

        elif child_type == bs4.element.Comment:
            continue
        elif child_type == bs4.element.Tag:

                if child.name in ignore_tags or (child.has_attr('class') and child['class'][0] == 'navigation'):
                    continue
                elif child.name == 'cite':
                    # add hrefs
                    hrefs = [a.get('href').strip('#') for a in child.find_all('a', class_='ltx_ref')]
                    local_text.append('~\cite{' + ', '.join(hrefs) + '}')
                elif child.name == 'img' and child.has_attr('alt'):
                    math_txt = child.get('alt')
                    if len(math_txt) < max_math_length:
                        local_text.append(math_txt)

                elif child.has_attr('class') and (child['class'][0] == 'ltx_Math' or child['class'][0] == 'ltx_equation'):
                    math_txt = child.get_text()
                    if len(math_txt) < max_math_length:
                        local_text.append(math_txt)

                elif child.name == 'section':
                    return
                else:
                    parse_text(local_text, child)
        else:
            raise RuntimeError('Unhandled type')

def clean_text(text):
    delete_items = ['=-1', '\t', u'\xa0', '[]', '()', 'mathbb', 'mathcal', 'bm', 'mathrm', 'mathit', 'mathbf', 'mathbfcal', 'textbf', 'textsc', 'langle', 'rangle', 'mathbin']
    for item in delete_items:
        text = text.replace(item, '')
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[[,]+]', '', text)
    text = re.sub(r'\.(?!\d)', '. ', text)
    text = re.sub('bib. bib', 'bib.bib', text)
    return text

def remove_stop_word_sections_and_extract_text(toc, soup, stop_words=['references', 'acknowledgments', 'about this document', 'apopendix']):
    def has_stop_word(title, stop_words):
        return any(stop_word.lower() in title.lower() for stop_word in stop_words)
    
    def extract_text(entry, soup):
        section_id = entry['id']
        if section_id: # section_id
            section = soup.find(id=section_id)
            if section is not None:
                local_text = []
                parse_text(local_text, section)
                if local_text:
                    processed_text = clean_text(''.join(local_text))
                    entry['text'] = processed_text
        return 0 
    
    def filter_and_update_toc(entries):
        filtered_entries = []
        for entry in entries:
            if not has_stop_word(entry['title'], stop_words):
                # Get clean text
                extract_text(entry, soup)                
                entry['subsections'] = filter_and_update_toc(entry['subsections'])
                filtered_entries.append(entry)
        return filtered_entries
    
    return filter_and_update_toc(toc)

def parse_html(html_file):
    soup = bs4.BeautifulSoup(html_file, "lxml")
    # parse title
    title = soup.head.title.get_text().replace("\n", " ")
    # parse abstract
    abstract = soup.find(class_='ltx_abstract').get_text()
    # parse citation
    citation = soup.find(class_='ltx_biblist')
    citation_dict = create_dict_for_citation(citation)
    # generate the full toc without text
    sections = generate_full_toc(soup)
    # remove the sections need to skip and extract the text of the rest sections
    sections = remove_stop_word_sections_and_extract_text(sections, soup)
    document = {
        "title": title, 
        "abstract": abstract, 
        "sections": sections, 
        "references": citation_dict,
    }
    return document 

def search_section_by_arxiv_id(entry_id, cite):
    warnings.warn("Using search_section_by_arxiv_id function may return wrong title because ar5iv parsing citation error. To solve this, You can prompt any LLM to extract the paper title from the reference string")
    assert re.match(r'^\d+\.\d+$', entry_id)
    url = f'https://ar5iv.labs.arxiv.org/html/{entry_id}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
            if not 'https://ar5iv.labs.arxiv.org/html' in html_content:
                warnings.warn(f'Invalid ar5iv HTML document: {url}')
                return None
            else:
                try:
                    document = parse_html(html_content)
                except:
                    warnings.warn(f'Wrong format HTML document: {url}')
                    return None
                try:
                    sections = get_2nd_section(document["sections"][0]["subsections"])
                except:
                    warnings.warn(f'Get subsections error')
                    return None
                sections2title = {}
                for k, v in sections.items():
                    k = " ".join(k.split("\n"))
                    sections2title[k] = set()
                    bibs = re.findall(cite, v, re.DOTALL)
                    for bib in bibs:
                        bib = bib.split(",")
                        for b in bib:
                            if b not in document["references"]:
                                continue
                            sections2title[k].add(document["references"][b]["title"]) # !!! The title here may be incorrect, you can use an LLM to parse the write title from document["references"][b]["meta_string"] !!!
                    if len(sections2title[k]) == 0:
                        del sections2title[k]
                    else:
                        sections2title[k] = list(sections2title[k])
                return sections2title
        else:
            warnings.warn(f"Failed to retrieve content. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        warnings.warn(f"An error occurred: {e}")
        return None

def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = ''.join(letters)
    return result.lower()

def search_paper_by_arxiv_id(arxiv_id):
    """
    Search paper by arxiv id.
    :param arxiv_id: arxiv id of the paper
    :return: paper list
    """
    if arxiv_id in id2paper:
        title_key = keep_letters(id2paper[arxiv_id])
        if title_key in paper_db.namelist():
            with paper_db.open(title_key) as f:
                data = json.loads(f.read().decode("utf-8"))
            return {
                "arxiv_id": arxiv_id,
                "title": data["title"].replace("\n", " "),
                "abstract": data["abstract"],
                "sections": data["sections"],
                "source": 'SearchFrom:local_paper_db',
            }

    search = arxiv.Search(
        query = "",
        id_list = [arxiv_id],
        max_results = 10,
        sort_by = arxiv.SortCriterion.Relevance,
        sort_order = arxiv.SortOrder.Descending,
    )

    try:
        results = list(arxiv_client.results(search, offset=0))
    except:
        warnings.warn(f"Failed to search arxiv id: {arxiv_id}")
        return None

    res = None
    for arxiv_id_result in results:
        entry_id = arxiv_id_result.entry_id.split("/")[-1]
        entry_id = entry_id.split('v')[0]
        if entry_id == arxiv_id:
            res = {
                "arxiv_id": arxiv_id,
                "title": arxiv_id_result.title.replace("\n", " "),
                "abstract": arxiv_id_result.summary.replace("\n", " "),
                "sections": "",
                "source": 'SearchFrom:arxiv',
            }
            break
    return res
    
def search_arxiv_id_by_title(title):
    """
    Search arxiv id by title.
    :param title: title of the paper
    :return: arxiv id of the paper
    """
    url = "https://arxiv.org/search/?" + urllib.parse.urlencode({
        'query': title,
        'searchtype': 'title', 
        'abstracts': 'hide', 
        'size': 200, 
    })
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
            soup = bs4.BeautifulSoup(html_content, 'html.parser')
            results = []

            if soup.find('meta', charset=True): # paper list
                if soup.find('p', class_="is-size-4 has-text-warning") and "Sorry" in soup.find('p', class_="is-size-4 has-text-warning").text.strip():
                    warnings.warn(f"Failed to find results by Arxiv Advanced Search: {title}")
                    return None
                
                p_tags = soup.find_all("li", class_="arxiv-result")
                for p_tag in p_tags:
                    title = p_tag.find("p", class_="title is-5 mathjax").text.strip()
                    id = p_tag.find('p', class_='list-title is-inline-block').find('a').text.strip('arXiv:')
                    if title and id:
                        results.append((title, id))
            if soup.find('html', xmlns=True): # a single paper
                p_tag = soup.find("head").find("title")
                match = re.match(r'\[(.*?)\]\s*(.*)', soup.title.string)
                if match:
                    id = match.group(1)
                    title = match.group(2)
                    if title and id:
                        results = [(title, id)]

            if results:
                for (result, id) in results:
                    title_find = result.lower().strip('.').replace(' ', '').replace('\n', '')
                    title_search = title.lower().strip('.').replace(' ', '').replace('\n', '')
                    if title_find == title_search:
                        return id
                return None
        
            warnings.warn(f"Failed to parse the html: {url}")
            return None
        else:
            warnings.warn(f"Failed to retrieve content. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        warnings.warn(f"An error occurred while search_arxiv_id_by_title: {e}")
        return None

def search_paper_by_title(title):
    """
    Search paper by title.
    :param title: title of the paper
    :return: paper list
    """
    title_id = search_arxiv_id_by_title(title)
    if title_id is None:
        return None
    title_id = title_id.split('v')[0]
    return search_paper_by_arxiv_id(title_id)

def get_subsection(sections):
    res = {}
    for section in sections:
        if "text" in section and section["text"].strip() != "":
            res[section["title"].strip()] = section["text"].strip()
        subsections = get_subsection(section["subsections"])
        for k, v in subsections.items():
            res[k] = v
    return res

def get_1st_section(sections):
    res = {}
    for section in sections:
        subsections = get_subsection(section["subsections"])
        if "text" in section and section["text"].strip() != "" or len(subsections) > 0:
            if "text" in section and section["text"].strip() != "":
                res[section["title"].strip()] = section["text"].strip()
            else:
                res[section["title"].strip()] = ""
            for k, v in subsections.items():
                res[section["title"].strip()] += v.strip()
    res_new = {}
    for k, v in res.items():
        if "appendix" not in k.lower():
            res_new[" ".join(k.split("\n")).strip()] = v
    return res_new

def get_2nd_section(sections):
    res = {}
    for section in sections:
        subsections = get_1st_section(section["subsections"])
        if "text" in section and section["text"].strip() != "":
            if "text" in section and section["text"].strip() != "":
                res[section["title"].strip()] = section["text"].strip()
        for k, v in subsections.items():
            res[section["title"].strip() + " " + k.strip()] = v.strip()
    res_new = {}
    for k, v in res.items():
        if "appendix" not in k.lower():
            res_new[" ".join(k.split("\n")).strip()] = v
    return res_new

def cal_micro(pred_set, label_set):
    if len(label_set) == 0:
        return 0, 0, 0

    if len(pred_set) == 0:
        return 0, 0, len(label_set)

    tp = len(pred_set & label_set)
    fp = len(pred_set - label_set)
    fn = len(label_set - pred_set)

    assert tp + fn == len(label_set)
    assert len(label_set) != 0
    return tp, fp, fn

if __name__ == "__main__":
    print(search_section_by_arxiv_id("2307.00235", r"~\\cite\{(.*?)\}"))
    # print(search_paper_by_arxiv_id("2307.00235"))
    # print(search_paper_by_title("A hybrid approach to CMB lensing reconstruction on all-sky intensity maps"))
