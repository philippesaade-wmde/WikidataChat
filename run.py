# -*- coding: utf-8 -*-
"""WikidataSerpAPIBeautifulSoupBloom.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14OtASMUAmKrxKxoncELKUZNF_IqEdwtH

https://www.wikidata.org/wiki/Q42395533
https://doc.wikimedia.org/Wikibase/master/js/rest-api/

https://www.wikidata.org/wiki/Wikidata:REST_API/Authentication
https://www.wikidata.org/wiki/Wikidata:REST_API

https://www.wikidata.org/w/rest.php/wikibase/v0/entities/items/Q42
https://www.wikidata.org/w/rest.php/wikibase/v0/entities/properties/P21
"""


from wikidatachat.textification import get_wikidata_statements_from_query
from wikidatachat.rag import question_answer_pipeline

if __name__ == '__main__':

    SERAPI_API_KEY = os.environ.get('SERAPI_API_KEY')
    HF_TOKEN = os.environ.get('HF_TOKEN')

    # global master_cache
    global item_cache
    global property_cache

    # master_cache = {}
    item_cache = {}
    property_cache = {}

    question = 'Who is Albert Einstein?'
    max_length = 2048
    n_sequences = 1
    model_name = "bigscience/bloom-560m"
    # model_name = "bigscience/bloom-7b1"
    # model_name = "bigscience/bloomz-7b1"
    serapi_api_key = SERAPI_API_KEY

    lang = 'en'
    quantization = False
    timeout = 100
    n_cores = cpu_count()
    api_url = 'https://www.wikidata.org/w'
    wikidata_base = '"wikidata.org"'
    verbose = True

    if 'text_input' not in locals():
        text_input = None

    if text_input is None:
        text_input = get_wikidata_statements_from_query(  # Textification
            question=question,
            lang=lang,
            timeout=timeout,
            n_cores=n_cores,
            wikidata_base=wikidata_base,
            serapi_api_key=serapi_api_key,
            verbose=verbose
        )

    question_answer_pipeline(
        question=question,
        max_length=max_length,
        n_sequences=n_sequences,
        model_name=model_name,
        serapi_api_key=serapi_api_key,
        lang=lang,
        quantization=quantization,
        timeout=timeout,
        n_cores=n_cores,
        text_input=text_input,
        api_url=api_url,
        wikidata_base=wikidata_base,
        verbose=verbose
    )

    """
    urls = ['https://www.wikidata.org/wiki/Q42']
    items_json = download_and_extract_items(
        urls=urls,
        wiki_url='https://www.wikidata.org/wiki',
        lang='en',
        timeout=100
    )

    items_json[0]['item_data']['labels']['en']

    property_cache.keys()
    item_cache.keys()

    # %timeit
    if len(items_json):
        statements = convert_wikidata_item_to_statements(items_json[0])
        print(f'{len(item_cache.keys())=}')
        print(f'{len(property_cache.keys())=}')
    else:
        print('items_json is empty')

    print(statements)
    """
