# Extract case details where reference is "SP-4"
state={}
state["extracted_data"]=[
    {
        "case_id": 4313330, 'reference': 'SPI9-2', 
        'created_date': '2025-02-19 05: 51: 11', 
        'phone': '25895471', 
        'title': 'nothing', 
        'form_domain': 'IT', 
        'form_category': 'N/A', 
        'priority': 'High', 
        'site': {'name': '3', 'id': 'IND012'
        }, 
        'critical': 0, 
        'escalated': 0, 
        'full_path': {'en': 'New Case', 'fr': 'Votre Demande'
        }, 
        'validation_status': {'total': 'N/A', 'accepted': 'N/A', 'refused': 'N/A', 'status': []
        }, 
        'claimant': {'name': 'S9_LASTNAME', 'email': 'norep.com'
        }, 
        'involved_users': [
            {'name': 'UAME', 'email': 'noreply@.com'
            }
        ], 
        'validators': [], 'approving_manager': {}, 'latest_action': {}, 'latest_status': {}, 'latest_graph_action': {}, 'actions': 'No actions available.'
    },
    {'case_id': 4313370, 
    'reference': 'SP-4', 
    'created_date': '2025-02-26 09: 18: 04', 
    'phone': '15935746259', 
    'title': 'GPU needed', 
    'form_domain': 'IT', 
    'form_category': 'IT Accessories', 
    'priority': 'High', 
    'site': {'name': 'us', 'id': 'IN12'
        }, 
        'critical': 'N/A', 
        'escalated': 'N/A', 
        'full_path': {'en': 'New Case', 'fr': 'Votre Demande'
        }, 
        'validation_status': {'total': 'N/A', 'accepted': 'N/A', 'refused': 'N/A', 'status': []
        }, 
        'claimant': {'name': 'SPME', 'email': 'nos.com'
        }, 
        'involved_users': [], 'validators': [], 'approving_manager': {}, 'latest_action': {}, 'latest_status': {}, 'latest_graph_action': {}, 'actions': 'No actions available.'
    }
]


# def match_ticket(user_query, state):
#     """Use fuzzy matching first, then apply cosine similarity"""
#     from fuzzywuzzy import process
#     from sentence_transformers import SentenceTransformer
#     from sklearn.metrics.pairwise import cosine_similarity

#     # Extract ticket titles
#     titles = [case["title"] for case in state["extracted_data"]]

#     # 1️⃣ First, Use Fuzzy Matching (Handle Typos)
#     best_match, score = process.extractOne(user_query, titles)

#     # 2️⃣ Then, Use Cosine Similarity (Semantic Search)
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     query_embedding = model.encode(user_query, normalize_embeddings=True)
#     title_embeddings = model.encode(titles, normalize_embeddings=True)

#     # Compute similarity
#     similarities = cosine_similarity([query_embedding], title_embeddings)[0]

#     # Get best semantic match
#     best_index = similarities.argmax()
#     best_semantic_match = titles[best_index]

#     print(f"🔍 Fuzzy Best Match: {best_match} (Score: {score})")
#     print(f"🤖 Cosine Best Match: {best_semantic_match} (Similarity: {similarities[best_index]:.2f})")

#     return best_match if score > 90 else best_semantic_match  # Pick the best result

# query= " abc issue"
# best_amtch=match_ticket(query,state)

import re
from fuzzywuzzy import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_query(user_query):
    """Cleans the user query by removing special characters & extra spaces."""
    user_query = re.sub(r"\b[a-zA-Z0-9]+-\w+-\w+\b", "", user_query)  # Remove ticket IDs
    user_query = re.sub(r"[^a-zA-Z0-9\s]", "", user_query)  # Remove punctuation
    return user_query.strip().lower()

def combined_similarity(user_query, state):
    """Combines FuzzyWuzzy (with python-Levenshtein) & Cosine Similarity for best ticket matching."""
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_cleaned = preprocess_query(user_query)

    # Extract ticket titles
    titles = [case["title"] for case in state["extracted_data"]]

    # 1️⃣ **Fuzzy Matching (Handles Typos & Word Reordering)**
    fuzzy_match, fuzzy_score = process.extractOne(query_cleaned, titles, scorer=fuzz.ratio)

    # 2️⃣ **Cosine Similarity (Understands Context)**
    query_embedding = model.encode(query_cleaned, normalize_embeddings=True)
    title_embeddings = model.encode(titles, normalize_embeddings=True)
    cosine_similarities = cosine_similarity([query_embedding], title_embeddings)[0]

    best_cosine_index = cosine_similarities.argmax()
    best_cosine_match = titles[best_cosine_index]
    cosine_score = cosine_similarities[best_cosine_index] * 100  # Convert to percentage scale

    # 3️⃣ **Blended Score Calculation (Weighted Combination)**
    alpha = 0.6  # Adjust weight (higher = favor Cosine, lower = favor Fuzzy)
    final_scores = [(alpha * cosine_similarities[i] * 100) + ((1 - alpha) * fuzz.ratio(query_cleaned, titles[i]))
                    for i in range(len(titles))]
    # print([x for x in final_scores])
    if max(final_scores)<30:
        print("no optimal result found")
        return

    best_combined_index = final_scores.index(max(final_scores))
    best_combined_match = titles[best_combined_index]

    # Get the best ticket ID
    best_ticket = state["extracted_data"][best_combined_index]
    ticket_id = best_ticket["case_id"]

    print(f"🔍 Fuzzy Best Match: {fuzzy_match} (Score: {fuzzy_score})")
    print(f"🤖 Cosine Best Match: {best_cosine_match} (Score: {cosine_score:.2f})")
    print(f"✨ Hybrid Best Match: {best_combined_match} (Final Score: {max(final_scores):.2f})")
    print(f"🎫 Best Ticket ID: {ticket_id}")

    return {"ticket_id": ticket_id, "best_match": best_combined_match}


# 🔥 Example Usage
user_query = " i have to type something here GPU needed, alspo here so that i can try to slow it down and check if its woing or not"
result = combined_similarity(user_query, state)

print("\n✅ Final Output:", result)
