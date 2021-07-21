from PIL import Image
import requests
import bs4

url = 'https://www.themoviedb.org/movie/862'

response = requests.get(url)

soup = bs4.BeautifulSoup(response.text, 'html.parser')

image = soup.find_all('overview')
# image_url = image['src']

print(image)
# print(image_url)

# img = Image.open(requests.get(image_url, stream = True).raw)
# img = Image.open(img).convert('RGB')
# img.save('resources/imgs/image.png')