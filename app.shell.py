from tiktoken import get_encoding, encoding_for_model
from weaviate_interface import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series
from openai_interface import GPT_Turbo
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_data)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
weaviate_api_key = os.environ['WEAVIATE_API_KEY']
weaviate_url = os.environ['WEAVIATE_ENDPOINT']
openai_api_key = os.environ['OPENAI_API_KEY']
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
##############
# START CODE #
##############
data_path = './data/impact_theory_data.json'
## RETRIEVER
client = WeaviateClient(weaviate_api_key, weaviate_url)
logger.info(f"client is live: {client.is_live()}, client is ready: {client.is_ready()}")
available_classes = sorted(client.show_classes())
logger.info(available_classes)

## RERANKER
reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

## LLM 
model_name = 'gpt-3.5-turbo-0613'
llm = GPT_Turbo(model=model_name, api_key=openai_api_key)

## ENCODING
encodings = encoding_for_model(model_name)

## INDEX NAME
index_name = 'Impact_theory_minilm_256'

##############
#  END CODE  #
##############
data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
        
    with st.sidebar:
        guest_input = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')

        alpha_input = st.slider('Alpha for Hybrid Search', 0.00, 1.00, step=0.45)
        retrieval_limit = st.slider('Limit for retrieval results', 1, 100, 10)
        reranker_topk = st.slider('Top K for Reranker', 1, 50, 3)
        temperature_input = st.slider('Temperature for LLM', 0.0, 2.0, 1.0)
        default_ix = available_classes.index("ImpactTheoryMinilm256")
        class_name = st.selectbox('Class Name:', options=available_classes, index=default_ix, placeholder='Select Class Name')

    client = WeaviateClient(weaviate_api_key, weaviate_url)

    # the original client class does not include 'summary'. 
    client.display_properties.append('summary')

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:
            ##############
            # START CODE #
            ##############
            guest_filter = WhereFilter(path=['guest'], operator='Equal', valueText=guest_input).todict() if guest_input else None

            hybrid_response = client.hybrid_search(query,
                                        class_name=class_name,
                                        alpha=alpha_input,
                                        # display_properties=display_properties,
                                        display_properties=client.display_properties,
                                        where_filter=guest_filter,
                                        limit=retrieval_limit)

            # rerank
            ranked_response = reranker.rerank(hybrid_response,
                                    query,
                                    apply_sigmoid=True,
                                    top_k=reranker_topk)

            # validate token count
            valid_response = validate_token_threshold(ranked_response, 
                                                       question_answering_prompt_series, 
                                                       query=query,
                                                       tokenizer= encodings, # variable from ENCODING,
                                                       token_threshold=4000, 
                                                       verbose=True)
            ##############
            #  END CODE  #
            ##############

            # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response)

            # prep for streaming response
            st.subheader("Response from Impact Theory")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                #creates container for LLM response
                chat_container, response_box = [], st.empty()
                
                # execute chat call to LLM
                ##############
                # START CODE #
                ##############
                for resp in llm.get_chat_completion(prompt=prompt,
                                    temperature=temperature_input,
                                    max_tokens=350,
                                    show_response=True,
                                    stream=True):
                    try:
                        with response_box:
                            content = resp.choices[0].delta.content
                            if content:
                                chat_container.append(content)
                                result = "".join(chat_container).strip()
                                st.write(f'{result}')
                    except Exception as e:
                        print(e)
                        continue
                ##############
                #  END CODE  #
                ##############
            ##############
            # START CODE #
            ##############
            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                col1, col2 = st.columns([7, 3], gap='large')
                image = hit['thumbnail_url']  # get thumbnail_url
                episode_url = hit['episode_url']  # get episode_url
                title = hit['title']  # get title
                show_length = hit['length']  # get length
                time_string = convert_seconds(show_length)  # convert show_length to readable time string
            ##############
            #  END CODE  #
            ##############
                with col1:
                    st.write(search_result(i=i, 
                                           url=episode_url, 
                                           guest=hit['guest'], 
                                           title=title, 
                                           content=hit['content'], 
                                           length=time_string),
                            unsafe_allow_html=True)
                    st.write('\n\n')
                with col2:
                    # st.write(f"<a href={episode_url} <img src={image} width='200'></a>", 
                    #             unsafe_allow_html=True)
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()