import requests
from tqdm import tqdm


def get_emoji_list(fname: str, url: str):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 12

        with open(fname, "wb") as f:
            for chunk in tqdm(
                response.iter_content(chunk_size=chunk_size),
                total=int(total_size / chunk_size),
                unit="KB",
                desc="Downloading",
            ):
                f.write(chunk)
    else:
        print(f"Failed to download file. HTTP status code: {response.status_code}")


if __name__ == "__main__":
    get_emoji_list(
        "emojis.html", "https://www.unicode.org/emoji/charts/full-emoji-list.html"
    )
    print("saved")
