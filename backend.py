import requests
from config import SEMANTIC_SCHOLAR_KEY

def fetch_papers(query):
    """
    Fetch papers related to a query from the Semantic Scholar API.

    Args:
        query (str): The search query string.

    Returns:
        list: A list of URLs for the open access PDFs of the papers.
    """
    # Define the query parameters for the API request
    query_params = {
        'query': query,
        'limit': 10,  # Limit the number of results to 10
        'fields': 'referenceCount,citationCount,title,openAccessPdf'  # Specify the fields to retrieve
    }
    headers = {"x-api-key": SEMANTIC_SCHOLAR_KEY}  # Set the API key in the request headers

    # Semantic Scholar API endpoint for paper search
    paper_relevancy_searchURL = 'https://api.semanticscholar.org/graph/v1/paper/search'

    # Make the GET request to the API
    search_response = requests.get(paper_relevancy_searchURL, params=query_params, headers=headers)

    url_list = []  # Initialize an empty list to store the URLs

    # Check if the request was successful
    if search_response.status_code == 200:
        # Parse the JSON response to get the list of papers
        papers_data = search_response.json().get('data', [])

        # Iterate through the papers and extract the URL of the open access PDF
        for paper in papers_data:
            if (paper.get('openAccessPdf', {}) != None) and "https://" in paper.get('openAccessPdf', {}).get("url"):
                url_list.append(paper.get('openAccessPdf', {}).get("url"))

    return url_list  # Return the list of URLs