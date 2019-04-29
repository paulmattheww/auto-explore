import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd
import requests

def fetch_google_geocode(address, api_key=None, return_full_response=False):
    '''Uses a str address and an api_key for Google Maps (if under 2500
    records being transformed in 24 hours) and returns the geocode and
    normalized address information.  

        Example:
            api_key = 'API KEY NUMBER SEVEN'
            fetch_google_geocode('4150 E La Paloma Dr Tucson AZ 85718 USA',
                                api_key=api_key)

    ARGS:
    KWARGS:
    RETURNS:
    '''
    # set URL
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json?address={}".format(address)
    if api_key is not None:
        geocode_url = geocode_url + "&key={}".format(api_key)

    # Ping google for the reuslts, convert to JSON
    results = requests.get(geocode_url)
    results = results.json()

    # if there's no results or an error, return empty results.
    if len(results['results']) == 0:
        output = {
            "formatted_address" : None,
            "latitude": None,
            "longitude": None,
            "accuracy": None,
            "google_place_id": None,
            "type": None,
            "postcode": None
        }
    else:
        answer = results['results'][0]
        output = {
            "formatted_address" : answer.get('formatted_address'),
            "latitude": answer.get('geometry').get('location').get('lat'),
            "longitude": answer.get('geometry').get('location').get('lng'),
            "accuracy": answer.get('geometry').get('location_type'),
            "google_place_id": answer.get("place_id"),
            "type": ",".join(answer.get('types')),
            "postcode": ",".join([x['long_name'] for x in answer.get('address_components')
                                  if 'postal_code' in x.get('types')])
        }

    # Append some other details:
    output['input_string'] = address
    output['number_of_results'] = len(results['results'])
    output['status'] = results.get('status')

    if return_full_response is True:
        output['response'] = results

    return output



def fetch_universal_sentence_embeddings(messages, verbose=0):
    """Fetches universal sentence embeddings from Google's research paper
    https://arxiv.org/pdf/1803.11175.pdf on sentence embeddings.

        Example:
            embeddings = fetch_universal_sentence_embeddings(txt)

    ARGS:
    KWARGS:
    RETURNS:
    """
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(messages))
        embeddings = list()
        for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
            if verbose:
                print("Message: {}".format(messages[i]))
                print("Embedding size: {}".format(len(message_embedding)))
                message_embedding_snippet = ", ".join(
                    (str(x) for x in message_embedding[:3]))
                print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
            embeddings.append(message_embedding)
    return embeddings
