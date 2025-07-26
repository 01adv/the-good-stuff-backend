

# from bs4 import BeautifulSoup
# import requests
# import json

# def extract_content(url, subcategory):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     data = []

#     # Extract Introduction
#     heading = soup.find('h1').get_text(strip=True) if soup.find('h1') else ""
#     intro_paragraph = soup.find('p').get_text(strip=True)[:200] if soup.find('p') else ""
#     data.append({
#         "subcategory": subcategory,
#         "content_type": "introduction",
#         "title": heading,
#         "text": f"{heading}\n{intro_paragraph}",
#         "url": url,
#         "category": "saas_optimization"
#     })

#     # Extract Case Studies
#     case_study_section = soup.find_all(class_='wp-block-stackable-column')
#     for case in case_study_section:
#         title = case.find(class_='stk-block-heading__text').get_text(strip=True) if case.find(class_='stk-block-heading__text') else ""
#         summary = case.find(class_='stk-block-text__text').get_text(strip=True)[:200] if case.find(class_='stk-block-text__text') else ""
#         link = case.find('a')['href'] if case.find('a') else url
#         if title and summary:  # Ensure valid case study
#             data.append({
#                 "subcategory": subcategory,
#                 "content_type": "case_study",
#                 "title": title,
#                 "text": f"{title}\n{summary}",
#                 "url": link,
#                 "category": "saas_optimization"
#             })

#     # Extract Insights
#     # Extract Insights
#     insight_section = soup.find_all(class_='stk-block-posts__item')
#     for insight in insight_section:
#         title = insight.find(class_='stk-block-posts__title').find('a').get_text(strip=True) if insight.find(class_='stk-block-posts__title') else ""
#         summary = insight.find(class_='stk-block-posts__excerpt').find('p').get_text(strip=True)[:200] if insight.find(class_='stk-block-posts__excerpt') else ""
#         link = insight.find('a')['href'] if insight.find('a') else url
#         if title and summary:  # Ensure valid insight
#             data.append({
#                 "subcategory": subcategory,
#                 "content_type": "insight",
#                 "title": title,
#                 "text": f"{title}\n{summary}",
#                 "url": link,
#                 "category": "saas_optimization"
#             })

#     return data

# # Crawl all subcategories
# subcategories = [
#     ("https://thegood.com/increase-registration/", "increase_registration"),
#     ("https://thegood.com/improve-onboarding/", "improve_onboarding"),
#     ("https://thegood.com/monetize-free-users/", "monetize_free_users"),
#     ("https://thegood.com/improve-retention/", "improve_retention"),
#     ("https://thegood.com/increase-referrals/", "increase_referrals"),
#     ("https://thegood.com/mitigate-cancellations/", "mitigate_cancellations")
# ]

# all_data = []
# for url, subcategory in subcategories:
#     all_data.extend(extract_content(url, subcategory))

# # Save to JSON
# with open("structured_data.json", "w") as f:
#     json.dump(all_data, f, indent=2)


from bs4 import BeautifulSoup
import requests
import json
from itertools import islice

def extract_full_content(url):
    """Extract up to 500 words of text content from a given URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract all paragraphs and combine text
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs)
        # Split into words and take up to 500
        words = text.split()
        limited_text = ' '.join(islice(words, 500))
        return limited_text if limited_text else ""
    except Exception:
        return ""

def extract_content(url, subcategory):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    data = []

    # Extract Introduction
    heading = soup.find('h1').get_text(strip=True) if soup.find('h1') else ""
    intro_paragraph = soup.find('p').get_text(strip=True)[:200] if soup.find('p') else ""
    data.append({
        "subcategory": subcategory,
        "content_type": "introduction",
        "title": heading,
        "text": f"{heading}\n{intro_paragraph}",
        "url": url,
        "category": "saas_optimization",
        "content_description": ""  # No full content for introduction
    })

    # Extract Case Studies
    case_study_section = soup.find_all(class_='wp-block-stackable-column')
    for case in case_study_section:
        title = case.find(class_='stk-block-heading__text').get_text(strip=True) if case.find(class_='stk-block-heading__text') else ""
        summary = case.find(class_='stk-block-text__text').get_text(strip=True)[:200] if case.find(class_='stk-block-text__text') else ""
        link = case.find('a')['href'] if case.find('a') else url
        if title and summary:  # Ensure valid case study
            content_description = extract_full_content(link) if link.startswith("https://thegood.com/results/") else ""
            data.append({
                "subcategory": subcategory,
                "content_type": "case_study",
                "title": title,
                "text": f"{title}\n{summary}",
                "url": link,
                "category": "saas_optimization",
                "content_description": content_description
            })

    # Extract Insights
    insight_section = soup.find_all(class_='stk-block-posts__item')
    for insight in insight_section:
        title = insight.find(class_='stk-block-posts__title').find('a').get_text(strip=True) if insight.find(class_='stk-block-posts__title') else ""
        summary = insight.find(class_='stk-block-posts__excerpt').find('p').get_text(strip=True)[:200] if insight.find(class_='stk-block-posts__excerpt') else ""
        link = insight.find('a')['href'] if insight.find('a') else url
        if title and summary:  # Ensure valid insight
            content_description = extract_full_content(link) if link.startswith("https://thegood.com/insights/") else ""
            data.append({
                "subcategory": subcategory,
                "content_type": "insight",
                "title": title,
                "text": f"{title}\n{summary}",
                "url": link,
                "category": "saas_optimization",
                "content_description": content_description
            })

    return data

# Crawl all subcategories
subcategories = [
    ("https://thegood.com/increase-registration/", "increase_registration"),
    ("https://thegood.com/improve-onboarding/", "improve_onboarding"),
    ("https://thegood.com/monetize-free-users/", "monetize_free_users"),
    ("https://thegood.com/improve-retention/", "improve_retention"),
    ("https://thegood.com/increase-referrals/", "increase_referrals"),
    ("https://thegood.com/mitigate-cancellations/", "mitigate_cancellations")
]

all_data = []
for url, subcategory in subcategories:
    all_data.extend(extract_content(url, subcategory))

# Save to JSON
with open("structured_data3.json", "w") as f:
    json.dump(all_data, f, indent=2)