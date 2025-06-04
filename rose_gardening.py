import json
from serpapi import GoogleSearch
import webbrowser

ROSE_TYPES = [
    "Hybrid Tea",
    "Floribunda",
    "Grandiflora",
    "Miniature",
    "Climbing",
]

def main():
    api_key = input("Enter your SerpAPI key: ").strip()
    if not api_key:
        print("SerpAPI key is required.")
        return

    print("Choose a rose type:")
    for i, rose in enumerate(ROSE_TYPES, 1):
        print(f"{i}. {rose}")
    choice = input("Enter choice number: ").strip()
    try:
        rose = ROSE_TYPES[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    query = f"how to cultivate {rose} rose"
    print(f"Searching for cultivation methods: {query}")
    search = GoogleSearch({"q": query, "api_key": api_key})
    results = search.get_dict()

    organic = results.get("organic_results", [])
    if organic:
        top = organic[0]
        print("\nTop search result:")
        print(top.get("title"))
        print(top.get("link"))
        print(top.get("snippet"))

    print("\nFetching images...")
    image_search = GoogleSearch({"q": f"{rose} rose", "tbm": "isch", "api_key": api_key})
    image_results = image_search.get_dict()
    images = image_results.get("images_results", [])[:4]

    for idx, img in enumerate(images, 1):
        url = img.get("original") or img.get("thumbnail")
        print(f"Image {idx}: {url}")
        webbrowser.open(url)

if __name__ == "__main__":
    main()
