import requests


def download_file_from_google_drive(dest_url, destination):
    id = url_to_id(dest_url)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def url_to_id(url):
    x = url.split("/")
    return x[5]
#
#
# if __name__ == "__main__":
#     dest_url = 'https://drive.google.com/file/d/1y-TINhR3ll1GO9cyhmC0AGt0yVsfy6qC/view?usp=share_link'
#     destination = 'output/abc.jpg'
#     download_file_from_google_drive(dest_url, destination)
