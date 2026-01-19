from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

job_roles = {
    "Data Scientist": "python machine learning statistics data analysis",
    "Web Developer": "html css javascript frontend backend",
    "AI Engineer": "python deep learning neural networks ai",
    "Cybersecurity Analyst": "security networking hacking encryption",
    "Software Engineer": "java python algorithms system design"
}

print("🧠 AI Job Role Recommender — Day 97\n")

user_skills = input("Enter your skills (comma separated): ").lower()

documents = list(job_roles.values()) + [user_skills]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(documents)

scores = cosine_similarity(vectors[-1], vectors[:-1])[0]

ranked = sorted(zip(job_roles.keys(), scores), key=lambda x: x[1], reverse=True)

print("\n🎯 Recommended Job Roles:")
for role, score in ranked:
    print(f"{role} → Match Score: {score*100:.2f}%")
